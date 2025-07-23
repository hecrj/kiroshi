use futures::future::{BoxFuture, FutureExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::io;
use tokio::net;

use std::collections::HashMap;
use std::marker::PhantomData;

pub use std::convert::Infallible as Never;

const ADDRESS: &str = "127.0.0.1:9149";
pub const PORT: u64 = 9149;

#[derive(Default)]
pub struct Router {
    routes: HashMap<&'static str, Handler>,
}

struct Handler(
    Box<dyn Fn(net::TcpStream, Vec<u8>) -> BoxFuture<'static, io::Result<()>> + Send + Sync>,
);

impl Router {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn endpoint<Request, Response, F>(
        mut self,
        task: Task<Request, Response>,
        handler: impl Fn(Connection<Response, Request>) -> F + Send + Sync + 'static,
    ) -> Self
    where
        F: Future<Output = io::Result<()>> + Send + 'static,
    {
        self.routes.insert(
            task.name,
            Handler(Box::new(move |stream, buffer| {
                handler(Connection {
                    stream,
                    buffer,
                    _types: PhantomData,
                })
                .boxed()
            })),
        );

        self
    }

    pub async fn handle(&self, mut stream: net::TcpStream) -> io::Result<()> {
        let mut buffer = Vec::new();

        let route: String = read_json(&mut stream, &mut buffer).await?;

        if let Some(handler) = self.routes.get(route.as_str()) {
            (handler.0)(stream, buffer).await?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Task<Request, Response> {
    name: &'static str,
    _types: PhantomData<(Request, Response)>,
}

impl<Request, Response> Task<Request, Response> {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            _types: PhantomData,
        }
    }

    pub async fn start(self) -> io::Result<Connection<Request, Response>> {
        let mut stream = net::TcpStream::connect(ADDRESS).await?;
        send_json(&mut stream, self.name).await?;

        Ok(Connection::new_unsafe(stream))
    }
}

pub struct Connection<I, O> {
    stream: net::TcpStream,
    buffer: Vec<u8>,
    _types: PhantomData<(I, O)>,
}

impl<I, O> Connection<I, O> {
    pub fn new_unsafe(stream: net::TcpStream) -> Self {
        Self {
            stream,
            buffer: Vec::new(),
            _types: PhantomData,
        }
    }

    pub async fn write(&mut self, input: I) -> io::Result<()>
    where
        I: Serialize,
    {
        send_json(&mut self.stream, input).await
    }

    pub async fn read(&mut self) -> io::Result<O>
    where
        O: DeserializeOwned,
    {
        read_json(&mut self.stream, &mut self.buffer).await
    }

    pub async fn read_bytes(&mut self) -> io::Result<&[u8]> {
        let n = read_bytes(&mut self.stream, &mut self.buffer).await?;

        Ok(&self.buffer[..n])
    }

    pub async fn copy<T>(&mut self, from: &mut Connection<T, O>) -> io::Result<u64> {
        io::copy(&mut self.stream, &mut from.stream).await
    }
}

async fn read_bytes(stream: &mut net::TcpStream, buffer: &mut Vec<u8>) -> io::Result<usize> {
    use tokio::io::AsyncReadExt;

    let message_size = stream.read_u64().await? as usize;

    if buffer.len() < message_size {
        buffer.resize(message_size, 0);
    }

    stream.read_exact(&mut buffer[..message_size]).await
}

async fn read_json<T: DeserializeOwned>(
    stream: &mut net::TcpStream,
    buffer: &mut Vec<u8>,
) -> io::Result<T> {
    let message_size = read_bytes(stream, buffer).await?;
    let data = serde_json::from_reader(&buffer[..message_size])?;

    Ok(data)
}

async fn send_json<T: Serialize>(stream: &mut net::TcpStream, data: T) -> io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let bytes = serde_json::to_vec(&data)?;

    stream.write_u64(bytes.len() as u64).await?;
    stream.write_all(bytes.as_slice()).await?;
    stream.flush().await?;

    Ok(())
}
