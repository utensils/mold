use std::io;
use std::net::TcpListener;

/// Probe the configured loopback port without silently choosing another one.
pub(super) fn available(port: u16) -> io::Result<u16> {
    if port == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "server_port must be nonzero",
        ));
    }
    TcpListener::bind(("127.0.0.1", port))?
        .local_addr()
        .map(|addr| addr.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_available_configured_port_is_preserved() {
        let probe = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        assert_eq!(available(port).unwrap(), port);
    }

    #[test]
    fn an_occupied_port_fails_instead_of_selecting_a_random_port() {
        let occupied = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = occupied.local_addr().unwrap().port();
        assert_eq!(
            available(port).unwrap_err().kind(),
            io::ErrorKind::AddrInUse
        );
    }

    #[test]
    fn zero_is_not_a_configured_port() {
        assert_eq!(
            available(0).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
