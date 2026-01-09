use crate::keypoint::KeyPoint;
use ndarray::{Array3, Axis};

/// SIFT descriptor computer
pub struct SiftDescriptor {
  gaussian_sigma_factor: f32,
  num_bins: usize,
  peak_ratio: f32,
  descriptor_width: usize,
  descriptor_num_bins: usize,
}

impl SiftDescriptor {
    /// Create a new SIFT descriptor computer
    pub fn new() -> Self {
        SiftDescriptor {
          gaussian_sigma_factor: 1.5,
          num_bins: 36,
          peak_ratio: 0.8,
          descriptor_width: 4,
          descriptor_num_bins: 8,
        }
    }

    pub fn compute_descriptors(
        &self,
        keypoints: &mut [KeyPoint],
        scale_spaces: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>>
    ) {
        let mut gradient_magnitudes_pyramid: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>> = Vec::new();
        let mut gradient_orientations_pyramid: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>> = Vec::new();
        for (octave, scale_space) in scale_spaces.iter().enumerate() {
            let (num_scales, height, width) = scale_space.dim();
            let mut magnitudes_pyramid = Array3::<f32>::zeros((num_scales, height as usize, width as usize));
            let mut orientations_pyramid = Array3::<f32>::zeros((num_scales, height as usize, width as usize));
            for scale in 0..num_scales {
                let layer = scale_space.index_axis(Axis(0), scale);
                let (width, height) = layer.dim();
                for y in 1..height - 1 {
                    for x in 1..width - 1 {
                        // 使用中心差分计算梯度
                        let dx = layer[[x + 1, y]] - layer[[x - 1, y]];
                        let dy = layer[[x, y + 1]] - layer[[x, y - 1]];
                        
                        // 计算梯度幅值和方向
                        magnitudes_pyramid[[scale, y, x]] = (dx * dx + dy * dy).sqrt();
                        orientations_pyramid[[scale, y, x]] = dy.atan2(dx); // atan2(dy, dx) 得到 [-π, π]
                    }
                }
            }
            gradient_magnitudes_pyramid.push(magnitudes_pyramid);
            gradient_orientations_pyramid.push(orientations_pyramid);
        }

        for keypoint in keypoints.iter_mut() {
            // 检查关键点的scale是否在有效范围内
            let scale_idx = keypoint.scale as usize;
            if scale_idx >= gradient_magnitudes_pyramid[keypoint.octave].len_of(Axis(0)) {
                continue; 
            }
            
            keypoint.orientation = self.compute_keypoint_orientations(&keypoint, 
              gradient_magnitudes_pyramid[keypoint.octave].index_axis(Axis(0), scale_idx),
              gradient_orientations_pyramid[keypoint.octave].index_axis(Axis(0), scale_idx)
            );
        }

        for keypoint in keypoints.iter_mut() {
            // 检查关键点的scale是否在有效范围内
            let scale_idx = keypoint.scale as usize;
            if scale_idx >= gradient_magnitudes_pyramid[keypoint.octave].len_of(Axis(0)) {
                continue; 
            }
            
            keypoint.descriptor = self.compute_descriptor(keypoint,
                gradient_magnitudes_pyramid[keypoint.octave].index_axis(Axis(0), scale_idx),
                gradient_orientations_pyramid[keypoint.octave].index_axis(Axis(0), scale_idx));
        }
    }

    fn compute_keypoint_orientations(
        &self,
        kp: &KeyPoint,
        magnitudes_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        orientations_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
    ) -> f32 {
        let (width, height) = magnitudes_pyramid.dim();
        
        // 计算高斯权重窗口的半径
        let sigma = self.gaussian_sigma_factor * kp.scale;
        let radius = (3.0 * sigma).round() as i32;
        
        let mut histogram = vec![0.0; self.num_bins];
        
        // 在关键点周围的区域内构建方向直方图
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let region_x = (kp.x + dx as f32).round() as usize;
                let region_y = (kp.y + dy as f32).round() as usize;
                
                // 检查边界
                if region_x < 1 || region_x >= width - 1 || region_y < 1 || region_y >= height - 1 {
                    continue;
                }

                let magnitude = magnitudes_pyramid[[region_y, region_x]];
                let orientation = orientations_pyramid[[region_y, region_x]]; // atan2(dy, dx) 得到 [-π, π]
                
                // 调整方向：相对于关键点的方向
                let adjusted_orientation = orientation - kp.orientation;
                
                // 确保方向在 [0, 2π) 范围内
                let normalized_orientation = if adjusted_orientation < 0.0 {
                    adjusted_orientation + 2.0 * std::f32::consts::PI
                } else {
                    adjusted_orientation
                };
                
                // 计算高斯权重
                let weight = (-(dx * dx + dy * dy) as f32 / (2.0 * sigma * sigma)).exp();
                let weighted_magnitude = magnitude * weight;
                
                // 找到对应的bin
                let bin_f = normalized_orientation * self.num_bins as f32 / (2.0 * std::f32::consts::PI);
                let bin = bin_f as usize % self.num_bins;
                
                // 线性插值到相邻的bin
                let fraction = bin_f - bin as f32;
                histogram[bin] += weighted_magnitude * (1.0 - fraction);
                histogram[(bin + 1) % self.num_bins] += weighted_magnitude * fraction;
            }
        }
        
        // 平滑直方图
        let smoothed_histogram = self.smooth_histogram(&histogram);
        
        // 寻找峰值方向
        self.find_peak_orientations(&smoothed_histogram)
    }

    fn smooth_histogram(&self, histogram: &[f32]) -> Vec<f32> {
        let mut smoothed = vec![0.0; histogram.len()];
        for i in 0..histogram.len() {
            let prev = histogram[(i + histogram.len() - 1) % histogram.len()];
            let curr = histogram[i];
            let next = histogram[(i + 1) % histogram.len()];
            smoothed[i] = 0.25 * prev + 0.5 * curr + 0.25 * next;
        }
        smoothed
    }

    fn find_peak_orientations(&self, histogram: &[f32]) -> f32 {
        let mut orientations = 0.;
        
        // 找到全局最大值
        let max_value = histogram.iter().fold(0.0, |max: f32, &val| max.max(val));
        let threshold = max_value * self.peak_ratio;
        
        for i in 0..histogram.len() {
            let prev = histogram[(i + histogram.len() - 1) % histogram.len()];
            let curr = histogram[i];
            let next = histogram[(i + 1) % histogram.len()];
            
            // 检查是否是局部最大值且超过阈值
            if curr > prev && curr > next && curr >= threshold {
                // 抛物线插值精确方向
                let interpolated_bin = i as f32 + 0.5 * (prev - next) / (prev - 2.0 * curr + next);
                let orientation = interpolated_bin * 2.0 * std::f32::consts::PI / self.num_bins as f32;
                orientations = orientation;
            }
        }
        
        orientations
    }

    fn compute_descriptor(
        &self,
        kp: &KeyPoint,
        magnitudes_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        orientations_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
    ) -> Vec<f32> {
        let (width, height) = magnitudes_pyramid.dim();
        // 16x16邻域，对应4x4个cell，每个cell是4x4像素
        let descriptor_size = 16.0;
        let cell_size = descriptor_size / self.descriptor_width as f32; // 4.0
        
        // 旋转矩阵（用于方向不变性）
        let cos_theta = kp.orientation.cos();
        let sin_theta = kp.orientation.sin();
        
        let mut descriptor_vector = vec![0.0; self.descriptor_width * self.descriptor_width * self.descriptor_num_bins];
        
        // 高斯权重参数（σ = 窗口宽度的一半）
        let sigma = 0.5 * descriptor_size;
        let invsig2 = 1.0 / (2.0 * sigma * sigma);
        
        // 在16x16窗口内采样
        for dy in -8..8 {
            for dx in -8..8 {
                // 旋转到关键点方向
                let rx = (cos_theta * dx as f32 + sin_theta * dy as f32) / kp.scale;
                let ry = (-sin_theta * dx as f32 + cos_theta * dy as f32) / kp.scale;
                
                // 计算在原图上的坐标
                let x = kp.x + dx as f32;
                let y = kp.y + dy as f32;
                
                // 检查边界
                if x < 1.0 || y < 1.0 || x >= width as f32 - 1.0 || y >= height as f32 - 1.0 {
                    continue;
                }

                // 高斯窗口权重
                let win_weight = (-(dx * dx + dy * dy) as f32 * invsig2).exp();

                // 获取梯度幅度和方向
                let xi = x.round() as usize;
                let yi = y.round() as usize;
                let magnitude = magnitudes_pyramid[[yi, xi]] * win_weight;
                let mut ori = orientations_pyramid[[yi, xi]] - kp.orientation;
                
                // 确保ori在[0, 2π)范围内
                while ori < 0.0 {
                    ori += 2.0 * std::f32::consts::PI;
                }
                while ori >= 2.0 * std::f32::consts::PI {
                    ori -= 2.0 * std::f32::consts::PI;
                }

                // 映射到cell坐标和方向bin
                let cx = (rx / cell_size) + (self.descriptor_width as f32 / 2.0);
                let cy = (ry / cell_size) + (self.descriptor_width as f32 / 2.0);
                
                if cx < -1.0 || cx >= self.descriptor_width as f32 || 
                   cy < -1.0 || cy >= self.descriptor_width as f32 {
                    continue;
                }

                // 三线性插值
                let c0 = cx.max(-1.0).min(self.descriptor_width as f32) as i32;
                let r0 = cy.max(-1.0).min(self.descriptor_width as f32) as i32;
                let o0 = (ori * self.descriptor_num_bins as f32 / (2.0 * std::f32::consts::PI)) as i32;
                
                let dc = cx - c0 as f32;
                let dr = cy - r0 as f32;
                let do_ = (ori * self.descriptor_num_bins as f32 / (2.0 * std::f32::consts::PI)) - o0 as f32;
                
                // 分配到相邻的bins
                for (rr, wr) in [(r0, 1.0 - dr), (r0 + 1, dr)] {
                    if rr < 0 || rr >= self.descriptor_width as i32 { continue; }
                    for (cc, wc) in [(c0, 1.0 - dc), (c0 + 1, dc)] {
                        if cc < 0 || cc >= self.descriptor_width as i32 { continue; }
                        for (oo, wo) in [(o0, 1.0 - do_), (o0 + 1, do_)] {
                            let rridx = rr as usize;
                            let ccidx = cc as usize;
                            let ooidx = (oo + self.descriptor_num_bins as i32) % self.descriptor_num_bins as i32;
                            
                            if rridx < self.descriptor_width && ccidx < self.descriptor_width && ooidx >= 0 {
                                let idx = (rridx * self.descriptor_width + ccidx) * self.descriptor_num_bins + ooidx as usize;
                                if idx < descriptor_vector.len() {
                                    let contrib = magnitude * wr * wc * wo;
                                    descriptor_vector[idx] += contrib;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 归一化描述符
        self.normalize_descriptor(&mut descriptor_vector);
        
        descriptor_vector
    }

    fn normalize_descriptor(&self, descriptor: &mut [f32]) {
        // 计算L2范数
        let norm: f32 = descriptor.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm > std::f32::EPSILON {
            // 第一次归一化
            for value in descriptor.iter_mut() {
                *value /= norm;
            }
        }
        
        // 截断大值（增强对光照变化的鲁棒性）
        let max_value = 0.2;
        for value in descriptor.iter_mut() {
            if *value > max_value {
                *value = max_value;
            }
        }
        
        // 第二次归一化
        let norm_after_clipping: f32 = descriptor.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm_after_clipping > std::f32::EPSILON {
            for value in descriptor.iter_mut() {
                *value /= norm_after_clipping;
            }
        }
    }
}