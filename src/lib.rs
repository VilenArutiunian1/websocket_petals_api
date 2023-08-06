use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{connect_async, MaybeTlsStream};
use tokio_tungstenite::{tungstenite::protocol::Message, WebSocketStream};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    #[serde(rename = "meta-llama/Llama-2-70b-chat-hf")]
    Llama2_70bChatHf,
    #[serde(rename = "stabilityai/StableBeluga2")]
    StableBeluga2,
    #[serde(rename = "timdettmers/guanaco-65b")]
    Guanaco65b,
    #[serde(rename = "enoch/llama-65b-hf")]
    Llama65bHf,
    #[serde(rename = "bigscience/bloomz")]
    Bloomz,
}

#[derive(Serialize)]
struct OpenSessionRequest {
    #[serde(rename = "type")]
    request_type: RequestType,
    max_length: u32,
    model: Option<Model>,
}

#[derive(Serialize, Deserialize)]
struct GenerateRequest {
    #[serde(rename = "type")]
    request_type: RequestType,
    model: Option<Model>,
    max_length: Option<u32>,
    inputs: Option<String>,
    stop_sequence: Option<String>,
    do_sample: Option<bool>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    max_new_tokens: Option<u32>,
}

#[derive(Clone, Serialize, Deserialize)]
enum RequestType {
    #[serde(rename = "generate")]
    Generate,
    #[serde(rename = "open_inference_session")]
    OpenInferenceSession,
}

#[derive(Debug, Deserialize)]
pub struct Response {
    pub ok: bool,
    pub outputs: String,
    pub stop: bool,
}

#[derive(Debug)]
pub enum OpenInferenceSessionError {
    TungsteniteError(tokio_tungstenite::tungstenite::Error),
    ApiError { traceback: String },
}

pub struct InferenceSession {
    ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl InferenceSession {
    pub async fn open<U>(url: U, max_length: u32, model: Option<Model>) -> Result<Self, OpenInferenceSessionError>
    where
        U: IntoClientRequest + Unpin,
    {
        let (mut ws_stream, _) = connect_async(url)
            .await
            .map_err(|e| OpenInferenceSessionError::TungsteniteError(e))?;
        let request = OpenSessionRequest {
            request_type: RequestType::OpenInferenceSession,
            model,
            max_length,
        };
        let open_session_request_str = serde_json::to_string(&request).unwrap();
        ws_stream
            .send(Message::Text(open_session_request_str))
            .await
            .map_err(|e| OpenInferenceSessionError::TungsteniteError(e))?;
        let message = ws_stream.next().await.unwrap().unwrap();
        let message = message.to_text().unwrap();
        let response: Value = serde_json::from_str(message).unwrap();
        if response.get("ok").unwrap() == "false" {
            let traceback = response.get("traceback").unwrap().to_string();
            return Err(OpenInferenceSessionError::ApiError { traceback });
        }
        Ok(Self {
            ws_stream
        })
    }

    pub async fn generate(&mut self, params: GenerateParams) -> Result<(), tokio_tungstenite::tungstenite::Error> {
        let request = GenerateRequest {
            request_type: RequestType::Generate,
            model: params.model,
            max_length: params.max_length,
            inputs: params.inputs,
            stop_sequence: params.stop_sequence,
            do_sample: params.do_sample,
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            max_new_tokens: params.max_new_tokens,
        };
        let generate_request_str = serde_json::to_string(&request).unwrap();
        self.ws_stream.send(Message::Text(generate_request_str)).await?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct GenerateParams {
    model: Option<Model>,
    inputs: Option<String>,
    do_sample: Option<bool>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    max_length: Option<u32>,
    max_new_tokens: Option<u32>,
    stop_sequence: Option<String>,
}

pub struct GenerateParamsBuilder(GenerateParams);

impl GenerateParamsBuilder {
    pub fn new() -> Self {
        Self(GenerateParams {
            model: None,
            inputs: None,
            do_sample: None,
            temperature: None,
            top_k: None,
            top_p: None,
            max_length: None,
            max_new_tokens: None,
            stop_sequence: None,
        })
    }

    pub fn model(mut self, model: Model) -> Self {
        self.0.model = Some(model);
        self
    }

    pub fn inputs(mut self, inputs: String) -> Self {
        self.0.inputs = Some(inputs);
        self
    }

    pub fn do_sample(mut self, do_sample: bool) -> Self {
        self.0.do_sample = Some(do_sample);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.0.temperature = Some(temperature);
        self
    }

    pub fn top_k(mut self, top_k: u32) -> Self {
        self.0.top_k = Some(top_k);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.0.top_p = Some(top_p);
        self
    }

    pub fn max_length(mut self, max_length: u32) -> Self {
        self.0.max_length = Some(max_length);
        self
    }

    pub fn max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.0.max_new_tokens = Some(max_new_tokens);
        self
    }

    pub fn stop_sequence(mut self, stop_sequence: String) -> Self {
        self.0.stop_sequence = Some(stop_sequence);
        self
    }

    pub fn build(self) -> Option<GenerateParams> {
        if self.0.max_length.is_none() && self.0.max_new_tokens.is_none() {
            return None;
        }

        Some(self.0)
    }
}
