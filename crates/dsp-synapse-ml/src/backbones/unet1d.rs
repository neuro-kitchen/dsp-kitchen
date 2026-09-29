//! 1D U-Net Encoder-Decoder backbone with skip connections for waveform denoising
//! and overlapping spike separation.

use anyhow::Result;
use crate::backend::{SynapseMlDevice, Tensor3D};
use crate::hub::SafetensorsMap;
use super::conv1d_res::{Conv1dLayer, ResBlock1D};

/// 2-stage 1D U-Net Encoder-Decoder with residual blocks and channel-concatenated skip connections.
#[derive(Debug, Clone)]
pub struct UNet1DBackbone {
    pub enc1: ResBlock1D,
    pub enc2: ResBlock1D,
    pub bottleneck: ResBlock1D,
    pub dec2: ResBlock1D,
    pub dec1: ResBlock1D,
    pub head: Conv1dLayer,
}

impl UNet1DBackbone {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        base_channels: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let c1 = base_channels.max(4);
        let c2 = c1 * 2;
        let c_bot = c2 * 2;

        let enc1 = ResBlock1D::new(in_channels, c1, 3, seed, device);
        let enc2 = ResBlock1D::new(c1, c2, 3, seed.wrapping_add(1_000), device);
        let bottleneck = ResBlock1D::new(c2, c_bot, 3, seed.wrapping_add(2_000), device);

        // Decoder stage 2 concatenates upsampled bottleneck (c_bot) + enc2 skip (c2)
        let dec2 = ResBlock1D::new(c_bot + c2, c2, 3, seed.wrapping_add(3_000), device);
        // Decoder stage 1 concatenates upsampled dec2 (c2) + enc1 skip (c1)
        let dec1 = ResBlock1D::new(c2 + c1, c1, 3, seed.wrapping_add(4_000), device);

        let head = Conv1dLayer::new_initialized(
            c1,
            out_channels,
            1,
            1,
            0,
            seed.wrapping_add(5_000),
            device,
        );

        Self {
            enc1,
            enc2,
            bottleneck,
            dec2,
            dec1,
            head,
        }
    }

    /// Forward pass `[N, C_in, T] -> [N, C_out, T]`.
    pub fn forward(&self, input: &Tensor3D) -> Tensor3D {
        let orig_t = input.shape[2];

        // Encoder 1
        let s1 = self.enc1.forward(input); // [N, c1, T]
        let p1 = s1.max_pool1d(2, 2); // [N, c1, T/2]

        // Encoder 2
        let s2 = self.enc2.forward(&p1); // [N, c2, T/2]
        let p2 = s2.max_pool1d(2, 2); // [N, c2, T/4]

        // Bottleneck
        let b = self.bottleneck.forward(&p2); // [N, c_bot, T/4]

        // Decoder 2
        let up2 = b.upsample_nearest_1d(2).match_temporal_length(s2.shape[2]);
        let cat2 = up2.concat_channels(&s2);
        let d2 = self.dec2.forward(&cat2);

        // Decoder 1
        let up1 = d2.upsample_nearest_1d(2).match_temporal_length(s1.shape[2]);
        let cat1 = up1.concat_channels(&s1);
        let d1 = self.dec1.forward(&cat1);

        // Final 1x1 projection head matched to input temporal length
        self.head.forward(&d1).match_temporal_length(orig_t)
    }

    pub fn save_weights(&self, prefix: &str, map: &mut SafetensorsMap) {
        self.enc1.save_weights(&format!("{}.enc1", prefix), map);
        self.enc2.save_weights(&format!("{}.enc2", prefix), map);
        self.bottleneck
            .save_weights(&format!("{}.bottleneck", prefix), map);
        self.dec2.save_weights(&format!("{}.dec2", prefix), map);
        self.dec1.save_weights(&format!("{}.dec1", prefix), map);
        self.head.save_weights(&format!("{}.head", prefix), map);
    }

    pub fn load_weights(
        &mut self,
        prefix: &str,
        map: &SafetensorsMap,
        device: SynapseMlDevice,
    ) -> Result<()> {
        self.enc1
            .load_weights(&format!("{}.enc1", prefix), map, device)?;
        self.enc2
            .load_weights(&format!("{}.enc2", prefix), map, device)?;
        self.bottleneck
            .load_weights(&format!("{}.bottleneck", prefix), map, device)?;
        self.dec2
            .load_weights(&format!("{}.dec2", prefix), map, device)?;
        self.dec1
            .load_weights(&format!("{}.dec1", prefix), map, device)?;
        self.head
            .load_weights(&format!("{}.head", prefix), map, device)?;
        Ok(())
    }
}
