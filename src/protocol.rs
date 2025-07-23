use crate::Error;
use crate::image;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::net;

const ADDRESS: &str = "127.0.0.1:9149";
pub const PORT: u64 = 9149;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    Ping,
    ListModels,
    GenerateImage {
        definition: image::Definition,
        preview_after: Option<f32>,
    },
}

pub async fn connect() -> Result<net::TcpStream, Error> {
    Ok(net::TcpStream::connect(ADDRESS).await?)
}

pub async fn perform(request: Request) -> Result<net::TcpStream, Error> {
    let mut stream = connect().await?;

    send_json(&mut stream, request).await?;

    Ok(stream)
}

pub async fn ping() -> Result<(), Error> {
    let mut stream = perform(Request::Ping).await?;

    #[derive(Deserialize)]
    struct Response(bool);

    let mut buffer = Vec::new();
    let Response(_pong) = read_json(&mut stream, &mut buffer).await?;

    Ok(())
}

pub async fn read_bytes(stream: &mut net::TcpStream, buffer: &mut Vec<u8>) -> Result<usize, Error> {
    use tokio::io::AsyncReadExt;

    let message_size = stream.read_u64().await? as usize;

    if buffer.len() < message_size {
        buffer.resize(message_size, 0);
    }

    Ok(stream.read_exact(&mut buffer[..message_size]).await?)
}

pub async fn read_json<T: DeserializeOwned>(
    stream: &mut net::TcpStream,
    buffer: &mut Vec<u8>,
) -> Result<T, Error> {
    let message_size = read_bytes(stream, buffer).await?;
    let data = serde_json::from_reader(&buffer[..message_size])?;

    Ok(data)
}

pub async fn send_json<T: Serialize>(stream: &mut net::TcpStream, data: T) -> Result<(), Error> {
    use tokio::io::AsyncWriteExt;

    let bytes = serde_json::to_vec(&data)?;

    stream.write_u64(bytes.len() as u64).await?;
    stream.write_all(bytes.as_slice()).await?;
    stream.flush().await?;

    Ok(())
}
