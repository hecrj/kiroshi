use crate::Error;

use tokio::process;

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub vram_usage: Memory,
    pub gpu_temperature: Temperature,
}

impl Stats {
    pub async fn fetch() -> Result<Self, Error> {
        // TODO: Run on sever
        let (vram_usage, gpu_temperature) = check_gpu().await?;

        Ok(Self {
            vram_usage,
            gpu_temperature,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memory {
    free: MBytes,
    total: MBytes,
}

impl Memory {
    pub fn ratio(self) -> f32 {
        self.used().0 as f32 / self.total.0 as f32
    }

    pub fn used(self) -> MBytes {
        MBytes(self.total.0 - self.free.0)
    }

    pub fn total(self) -> MBytes {
        self.total
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Temperature {
    pub celsius: u64,
}

impl fmt::Display for Temperature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} °C", self.celsius)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MBytes(usize);

impl MBytes {
    pub fn from_mebibytes(mebibytes: usize) -> Self {
        Self(mebibytes * 1000 / 1024)
    }
}

impl fmt::Display for MBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} MB", self.0)
    }
}

async fn check_gpu() -> Result<(Memory, Temperature), Error> {
    let nvidia_smi = process::Command::new("nvidia-smi")
        .args(["-q", "-d", "MEMORY,TEMPERATURE"])
        .output()
        .await?;

    let unexpected_output = || Error::InvalidOutput("unexpected output by nvidia-smi".to_owned());

    let output = String::from_utf8(nvidia_smi.stdout).map_err(|_| unexpected_output())?;

    let total = output
        .lines()
        .find(|line| line.trim().starts_with("Total"))
        .and_then(|line| line.trim().split_whitespace().rev().nth(1))
        .ok_or(unexpected_output())?
        .parse()
        .map(MBytes::from_mebibytes)
        .ok()
        .ok_or(unexpected_output())?;

    let free = output
        .lines()
        .find(|line| line.trim().starts_with("Free"))
        .and_then(|line| line.trim().split_whitespace().rev().nth(1))
        .ok_or(unexpected_output())?
        .parse()
        .map(MBytes::from_mebibytes)
        .ok()
        .ok_or(unexpected_output())?;

    let memory = Memory { free, total };

    let temperature = output
        .lines()
        .find(|line| line.trim().starts_with("GPU Current Temp"))
        .and_then(|line| line.trim().split_whitespace().rev().nth(1))
        .ok_or(unexpected_output())?
        .parse()
        .map(|celsius| Temperature { celsius })
        .ok()
        .ok_or(unexpected_output())?;

    Ok((memory, temperature))
}
