use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};

static REQUEST_ID_HEADER: &str = "x-request-id";

/// Request-scoped audit identifier installed before authentication and route
/// handling.
#[derive(Clone, Debug)]
pub(crate) struct RequestId(pub(crate) String);

/// Axum middleware that ensures every request/response has an `X-Request-ID` header.
///
/// If the client sends one, it is preserved. Otherwise a UUID v4 is generated.
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_str(&uuid::Uuid::new_v4().to_string()).unwrap());
    let audit_id = id
        .to_str()
        .map(str::to_owned)
        .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
    request.extensions_mut().insert(RequestId(audit_id));

    let mut response = next.run(request).await;
    response.headers_mut().insert(REQUEST_ID_HEADER, id);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, middleware, routing::get, Router};
    use tower::ServiceExt;

    async fn ok_handler() -> &'static str {
        "ok"
    }

    #[tokio::test]
    async fn generates_request_id_when_missing() {
        let app = Router::new()
            .route("/", get(ok_handler))
            .layer(middleware::from_fn(request_id_middleware));

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(resp.headers().contains_key("x-request-id"));
        let val = resp
            .headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap();
        // Should be a valid UUID v4
        assert!(uuid::Uuid::parse_str(val).is_ok());
    }

    #[tokio::test]
    async fn preserves_existing_request_id() {
        let app = Router::new()
            .route("/", get(ok_handler))
            .layer(middleware::from_fn(request_id_middleware));

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/")
                    .header("x-request-id", "my-custom-id")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let val = resp
            .headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(val, "my-custom-id");
    }
}
