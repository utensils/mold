//! Uses documented Win32 access-token and security-descriptor APIs. Creation
//! supplies a protected DACL with exactly the process user's SID; validation
//! reads owner/DACL from the same open non-reparse handle used for key reads.
use super::invalid;
use std::{
    ffi::c_void,
    fs::File,
    io,
    mem::{size_of, zeroed},
    os::windows::{
        ffi::OsStrExt,
        io::{AsRawHandle, FromRawHandle},
    },
    path::Path,
    ptr::{null, null_mut},
};
#[cfg(test)]
use windows_sys::Win32::Security::SetFileSecurityW;
use windows_sys::Win32::{
    Foundation::{
        CloseHandle, LocalFree, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    },
    Security::{
        AclSizeInformation,
        Authorization::{
            ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
            GetSecurityInfo, SE_FILE_OBJECT,
        },
        EqualSid, GetAce, GetAclInformation, GetSecurityDescriptorControl, GetTokenInformation,
        TokenUser, ACCESS_ALLOWED_ACE, ACL, ACL_SIZE_INFORMATION, DACL_SECURITY_INFORMATION,
        OWNER_SECURITY_INFORMATION, PSID, SECURITY_ATTRIBUTES, SE_DACL_PROTECTED, TOKEN_QUERY,
        TOKEN_USER,
    },
    Storage::FileSystem::{
        CreateDirectoryW, CreateFileW, GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
        CREATE_NEW, FILE_ALL_ACCESS, FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_NORMAL,
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        FILE_READ_ATTRIBUTES, FILE_SHARE_READ, OPEN_EXISTING, READ_CONTROL,
    },
    System::Threading::{GetCurrentProcess, OpenProcessToken},
};
fn wide(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}
struct Allocation(*mut c_void);
impl Drop for Allocation {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                LocalFree(self.0);
            }
        }
    }
}
struct Handle(HANDLE);
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}
struct User(Vec<u64>);
impl User {
    fn current() -> io::Result<Self> {
        unsafe {
            let mut token = null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return Err(io::Error::last_os_error());
            }
            let token = Handle(token);
            let mut size = 0;
            GetTokenInformation(token.0, TokenUser, null_mut(), 0, &mut size);
            if size == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut data = vec![0_u64; (size as usize).div_ceil(8)];
            if GetTokenInformation(
                token.0,
                TokenUser,
                data.as_mut_ptr().cast(),
                size,
                &mut size,
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(Self(data))
        }
    }
    fn sid(&self) -> PSID {
        unsafe { (*(self.0.as_ptr().cast::<TOKEN_USER>())).User.Sid }
    }
    fn string(&self) -> io::Result<String> {
        unsafe {
            let mut value = null_mut();
            if ConvertSidToStringSidW(self.sid(), &mut value) == 0 {
                return Err(io::Error::last_os_error());
            }
            let _guard = Allocation(value.cast());
            let mut len = 0;
            while *value.add(len) != 0 {
                len += 1;
            }
            Ok(String::from_utf16_lossy(std::slice::from_raw_parts(
                value, len,
            )))
        }
    }
}
fn descriptor(user: &User, extra: &str) -> io::Result<Allocation> {
    let sid = user.string()?;
    let sddl = format!("O:{sid}D:P(A;;FA;;;{sid}){extra}");
    let wide = sddl.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    let mut descriptor = null_mut();
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            wide.as_ptr(),
            1,
            &mut descriptor,
            null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(Allocation(descriptor))
}
pub(super) fn create_directory(path: &Path) -> io::Result<()> {
    let user = User::current()?;
    let descriptor = descriptor(&user, "")?;
    let attributes = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor.0,
        bInheritHandle: 0,
    };
    if unsafe { CreateDirectoryW(wide(path).as_ptr(), &attributes) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
pub(super) fn create_file(path: &Path) -> io::Result<File> {
    let user = User::current()?;
    let descriptor = descriptor(&user, "")?;
    let attributes = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor.0,
        bInheritHandle: 0,
    };
    let handle = unsafe {
        CreateFileW(
            wide(path).as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            &attributes,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_handle(handle) })
}
pub(super) fn open(path: &Path, directory: bool) -> io::Result<File> {
    let rights = READ_CONTROL | FILE_READ_ATTRIBUTES | if directory { 0 } else { GENERIC_READ };
    let flags = FILE_FLAG_OPEN_REPARSE_POINT
        | if directory {
            FILE_FLAG_BACKUP_SEMANTICS
        } else {
            FILE_ATTRIBUTE_NORMAL
        };
    let handle = unsafe {
        CreateFileW(
            wide(path).as_ptr(),
            rights,
            FILE_SHARE_READ,
            null(),
            OPEN_EXISTING,
            flags,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_handle(handle) })
}
pub(super) fn validate(file: &File, directory: bool) -> io::Result<()> {
    unsafe {
        let handle = file.as_raw_handle();
        let mut info: BY_HANDLE_FILE_INFORMATION = zeroed();
        if GetFileInformationByHandle(handle, &mut info) == 0 {
            return Err(io::Error::last_os_error());
        }
        if info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0
            || (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0) != directory
            || (!directory && info.nNumberOfLinks != 1)
        {
            return Err(invalid(
                "host identity must be a regular non-reparse file with no additional hard links",
            ));
        }
        let mut owner = null_mut();
        let mut dacl: *mut ACL = null_mut();
        let mut security = null_mut();
        let status = GetSecurityInfo(
            handle,
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            null_mut(),
            &mut dacl,
            null_mut(),
            &mut security,
        );
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status as i32));
        }
        let _security = Allocation(security);
        let user = User::current()?;
        if owner.is_null() || dacl.is_null() || EqualSid(owner, user.sid()) == 0 {
            return Err(invalid(
                "host identity requires the process user as owner and a private DACL",
            ));
        }
        let mut control = 0;
        let mut revision = 0;
        if GetSecurityDescriptorControl(security, &mut control, &mut revision) == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut size: ACL_SIZE_INFORMATION = zeroed();
        if GetAclInformation(
            dacl,
            (&mut size as *mut ACL_SIZE_INFORMATION).cast(),
            size_of::<ACL_SIZE_INFORMATION>() as u32,
            AclSizeInformation,
        ) == 0
        {
            return Err(io::Error::last_os_error());
        }
        if control & SE_DACL_PROTECTED == 0 || size.AceCount != 1 {
            return Err(invalid(
                "host identity DACL must be protected and allow only the process user",
            ));
        }
        let mut ace = null_mut();
        if GetAce(dacl, 0, &mut ace) == 0 {
            return Err(io::Error::last_os_error());
        }
        let ace = &*(ace.cast::<ACCESS_ALLOWED_ACE>());
        // ACCESS_ALLOWED_ACE_TYPE is the documented type byte 0. The SID begins at
        // SidStart and is variable length, owned by the security descriptor above.
        if ace.Header.AceType != 0
            || ace.Header.AceFlags != 0
            || ace.Mask != FILE_ALL_ACCESS
            || EqualSid((&ace.SidStart as *const u32).cast_mut().cast(), user.sid()) == 0
        {
            return Err(invalid(
                "host identity ACL grants access outside the current user",
            ));
        }
        Ok(())
    }
}
#[cfg(test)]
pub(super) fn weaken_for_test(path: &Path) -> io::Result<()> {
    let user = User::current()?;
    let descriptor = descriptor(&user, "(A;;FA;;;WD)")?;
    if unsafe { SetFileSecurityW(wide(path).as_ptr(), DACL_SECURITY_INFORMATION, descriptor.0) }
        == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
