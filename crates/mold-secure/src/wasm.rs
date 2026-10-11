use crate::*;
use wasm_bindgen::prelude::*;
fn js(error: Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}
#[wasm_bindgen(js_name=Identity)]
pub struct WasmIdentity {
    inner: Option<Identity>,
}
#[wasm_bindgen(js_class=Identity)]
impl WasmIdentity {
    pub fn generate() -> std::result::Result<WasmIdentity, JsValue> {
        Ok(Self {
            inner: Some(Identity::generate().map_err(js)?),
        })
    }
    #[wasm_bindgen(js_name=fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> std::result::Result<WasmIdentity, JsValue> {
        Ok(Self {
            inner: Some(Identity::from_bytes(bytes).map_err(js)?),
        })
    }
    #[wasm_bindgen(js_name=toBytes)]
    pub fn to_bytes(&self) -> std::result::Result<Vec<u8>, JsValue> {
        Ok(self.identity()?.to_bytes())
    }
    #[wasm_bindgen(js_name=publicSigningKey)]
    pub fn signing_public(&self) -> std::result::Result<Vec<u8>, JsValue> {
        Ok(self.identity()?.signing_public().to_vec())
    }
    #[wasm_bindgen(js_name=publicNoiseKey)]
    pub fn noise_public(&self) -> std::result::Result<Vec<u8>, JsValue> {
        Ok(self.identity()?.noise_public().to_vec())
    }
    pub fn binding(&self) -> std::result::Result<Vec<u8>, JsValue> {
        Ok(self.identity()?.binding().to_vec())
    }
    pub fn sign(&self, bytes: &[u8]) -> std::result::Result<Vec<u8>, JsValue> {
        if bytes.len() > 65535 {
            return Err(js(Error::Limit));
        }
        Ok(self.identity()?.sign(bytes).to_vec())
    }
    pub fn close(&mut self) {
        self.inner = None;
    }
}
impl WasmIdentity {
    fn identity(&self) -> std::result::Result<&Identity, JsValue> {
        self.inner.as_ref().ok_or_else(|| js(Error::State))
    }
}
#[wasm_bindgen(js_name=Session)]
pub struct WasmSession {
    inner: Session,
}
#[wasm_bindgen(js_class=Session)]
impl WasmSession {
    pub fn create(
        identity: &WasmIdentity,
        config: &str,
    ) -> std::result::Result<WasmSession, JsValue> {
        if config.len() > 16384 {
            return Err(js(Error::Limit));
        }
        let config = serde_json::from_str(config).map_err(|_| js(Error::Input))?;
        Ok(Self {
            inner: Session::new(identity.identity()?, config).map_err(js)?,
        })
    }
    #[wasm_bindgen(js_name=handshakeWrite)]
    pub fn handshake_write(&mut self) -> std::result::Result<Vec<u8>, JsValue> {
        self.inner.handshake_write().map_err(js)
    }
    #[wasm_bindgen(js_name=handshakeRead)]
    pub fn handshake_read(&mut self, bytes: &[u8]) -> std::result::Result<(), JsValue> {
        self.inner.handshake_read(bytes).map_err(js)
    }
    #[wasm_bindgen(js_name=handshakeHash)]
    pub fn handshake_hash(&self) -> std::result::Result<Vec<u8>, JsValue> {
        self.inner.handshake_hash().map(|v| v.to_vec()).map_err(js)
    }
    pub fn established(&self) -> bool {
        self.inner.established()
    }
    pub fn seal(&mut self, bytes: &[u8]) -> std::result::Result<Vec<u8>, JsValue> {
        self.inner.seal(bytes).map_err(js)
    }
    pub fn open(&mut self, bytes: &[u8]) -> std::result::Result<Vec<u8>, JsValue> {
        self.inner.open(bytes).map_err(js)
    }
    pub fn close(&mut self) {
        self.inner.close();
    }
}
#[wasm_bindgen(js_name=namespace)]
pub fn wasm_namespace(public: &[u8]) -> std::result::Result<String, JsValue> {
    if public.len() != 32 {
        return Err(js(Error::Input));
    }
    Ok(namespace(public))
}
#[wasm_bindgen(js_name=verifySignature)]
pub fn wasm_verify_signature(public: &[u8], message: &[u8], signature: &[u8]) -> bool {
    verify_signature(public, message, signature)
}
#[wasm_bindgen(js_name=DecodedRecord)]
pub struct DecodedRecord {
    inner: Record,
}
#[wasm_bindgen(js_class=DecodedRecord)]
impl DecodedRecord {
    pub fn kind(&self) -> u8 {
        self.inner.kind()
    }
    #[wasm_bindgen(js_name=streamId)]
    pub fn stream_id(&self) -> u32 {
        self.inner.stream_id()
    }
    pub fn payload(&self) -> Vec<u8> {
        self.inner.payload().unwrap_or_default()
    }
}
#[wasm_bindgen(js_name=decodeRecord)]
pub fn wasm_decode_record(bytes: &[u8]) -> std::result::Result<DecodedRecord, JsValue> {
    Ok(DecodedRecord {
        inner: Record::decode(bytes).map_err(js)?,
    })
}
#[wasm_bindgen(js_name=encodeRecord)]
pub fn wasm_encode_record(
    kind: u8,
    stream_id: u32,
    payload: &[u8],
) -> std::result::Result<Vec<u8>, JsValue> {
    if payload.len() + 9 > MAX_PLAINTEXT {
        return Err(js(Error::Limit));
    }
    let mut bytes = vec![kind];
    bytes.extend_from_slice(&stream_id.to_be_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    bytes.extend_from_slice(payload);
    Record::decode(&bytes).map_err(js)?;
    Ok(bytes)
}
