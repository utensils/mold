use anyhow::Error;

use crate::client::MoldClient;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerAvailability {
    FallbackLocal,
    SurfaceError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateServerAction {
    PullModelAndRetry,
    FallbackLocal,
    SurfaceError,
}

pub fn classify_server_error(err: &Error) -> ServerAvailability {
    if MoldClient::is_connection_error(err) {
        ServerAvailability::FallbackLocal
    } else {
        ServerAvailability::SurfaceError
    }
}

pub fn classify_generate_error(err: &Error) -> GenerateServerAction {
    if MoldClient::is_model_not_found(err) {
        GenerateServerAction::PullModelAndRetry
    } else if MoldClient::is_connection_error(err) {
        GenerateServerAction::FallbackLocal
    } else {
        GenerateServerAction::SurfaceError
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MoldError;

    #[test]
    fn classify_server_error_connection_falls_back_local() {
        let err = anyhow::Error::new(MoldError::Client("connect failed".into()));
        assert_eq!(
            classify_server_error(&err),
            ServerAvailability::FallbackLocal
        );
    }

    #[test]
    fn classify_server_error_non_connection_surfaces() {
        let err = anyhow::anyhow!("boom");
        assert_eq!(
            classify_server_error(&err),
            ServerAvailability::SurfaceError
        );
    }

    #[test]
    fn classify_generate_error_model_missing_pulls_and_retries() {
        let err = anyhow::Error::new(MoldError::ModelNotFound("missing".into()));
        assert_eq!(
            classify_generate_error(&err),
            GenerateServerAction::PullModelAndRetry
        );
    }

    #[test]
    fn classify_generate_error_connection_falls_back_local() {
        let err = anyhow::Error::new(MoldError::Client("connect failed".into()));
        assert_eq!(
            classify_generate_error(&err),
            GenerateServerAction::FallbackLocal
        );
    }

    #[test]
    fn classify_generate_error_other_surfaces() {
        let err = anyhow::anyhow!("boom");
        assert_eq!(
            classify_generate_error(&err),
            GenerateServerAction::SurfaceError
        );
    }
}
