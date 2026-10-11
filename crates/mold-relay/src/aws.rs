//! Credential-free Lambda endpoint validation. Data uses secure protocol3.
use super::*;
pub fn validate_aws_endpoint(endpoint: &str, allow_loopback_ws: bool) -> Result<url::Url> {
    let url =
        url::Url::parse(endpoint).map_err(|_| anyhow::anyhow!("invalid AWS relay endpoint"))?;
    let mut origin = url.clone();
    origin.set_path("/");
    validate_endpoint(origin.as_str(), allow_loopback_ws)?;
    if url.query().is_some() || url.fragment().is_some() {
        bail!("AWS relay endpoint must not contain query credentials");
    }
    Ok(url)
}
pub fn validate_host_namespace(host_id: &str) -> Result<()> {
    mold_secure::decode_hex::<16>(host_id)?;
    Ok(())
}
