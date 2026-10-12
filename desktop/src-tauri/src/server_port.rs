/// Reserve a port on `bind` by binding to :0 and reading the assignment.
///
/// TOCTOU caveat: the listener is dropped before `run_server` rebinds it.
/// Upstream fix tracked as U1 (`run_server_with_listener`) in
/// desktop/docs/architecture.md.
pub fn allocate_port(bind: &str) -> std::io::Result<u16> {
    let probe = std::net::TcpListener::bind((bind, 0))?;
    Ok(probe.local_addr()?.port())
}

/// Refuse an occupied preferred port. A failed ownership probe can mean a
/// slow existing Mold host; it must never cause a second engine on another port.
pub fn available_server_port_for(bind: &str, preferred: u16) -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind((bind, preferred)).map_err(|error| {
        std::io::Error::new(error.kind(), format!(
            "Cannot start This Mac on port {preferred}: another service may already own it. Wait for the existing Mold host to answer or stop it before starting this engine."
        ))
    })?;
    drop(listener);
    Ok(preferred)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocates_distinct_nonzero_ports() {
        let a = allocate_port("127.0.0.1").unwrap();
        let b = allocate_port("127.0.0.1").unwrap();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        // Not guaranteed distinct by the OS, but freshly-released ephemeral
        // ports are not immediately reused on macOS/Linux.
        assert_ne!(a, b);
    }

    #[test]
    fn prefers_well_known_port_when_available() {
        let preferred = allocate_port("127.0.0.1").unwrap();
        let port = available_server_port_for("127.0.0.1", preferred).unwrap();
        assert_eq!(port, preferred);
    }

    #[test]
    fn refuses_a_second_engine_when_well_known_port_is_occupied() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let occupied = listener.local_addr().unwrap().port();
        assert!(available_server_port_for("127.0.0.1", occupied).is_err());
        drop(listener);
    }
}
