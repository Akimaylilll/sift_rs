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
        keypoints: &mut Vec<KeyPoint>,
        scale_spaces: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>>
    ) {
        let mut gradient_magnitudes_pyramid: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>> = Vec::new();
        let mut gradient_orientations_pyramid: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>> = Vec::new();
        
        for (_octave, scale_space) in scale_spaces.iter().enumerate() {
            let (num_scales, height, width) = scale_space.dim();
            let mut magnitudes_pyramid = Array3::<f32>::zeros((num_scales, height as usize, width as usize));
            let mut orientations_pyramid = Array3::<f32>::zeros((num_scales, height as usize, width as usize));
            
            for scale in 0..num_scales {
                let layer = scale_space.index_axis(Axis(0), scale);
                let (img_width, img_height) = (layer.shape()[1], layer.shape()[0]); // width是第1轴，height是第0轴
                
                for y in 1..img_height - 1 {
                    for x in 1..img_width - 1 {
                        // 使用中心差分计算梯度
                        let dx = layer[[y, x + 1]] - layer[[y, x - 1]];  // x方向梯度
                        let dy = layer[[y + 1, x]] - layer[[y - 1, x]];  // y方向梯度
                        
                        // 计算梯度幅值和方向
                        magnitudes_pyramid[[scale, y, x]] = (dx * dx + dy * dy).sqrt();
                        let mut orientation = dy.atan2(dx); // atan2(dy, dx) 得到 [-π, π]
                        
                        // 将角度标准化到 [0, 2π]
                        if orientation < 0.0 {
                            orientation += 2.0 * std::f32::consts::PI;
                        }
                        
                        orientations_pyramid[[scale, y, x]] = orientation;
                    }
                }
            }
            gradient_magnitudes_pyramid.push(magnitudes_pyramid);
            gradient_orientations_pyramid.push(orientations_pyramid);
        }

        let mut all_keypoints: Vec<KeyPoint> = Vec::new();
        for keypoint in keypoints.drain(..) {
            // 检查关键点的octave和scale是否在有效范围内
            if keypoint.octave >= gradient_magnitudes_pyramid.len() {
                continue;
            }
            
            let octave_data = &gradient_magnitudes_pyramid[keypoint.octave];
            let octave_orientation = &gradient_orientations_pyramid[keypoint.octave];
            
            if keypoint.scale as usize >= octave_data.len_of(Axis(0)) {
                continue;
            }
            
            let scale_idx = keypoint.scale as usize;
            let magnitude_layer = octave_data.index_axis(Axis(0), scale_idx);
            let orientation_layer = octave_orientation.index_axis(Axis(0), scale_idx);
            
            let orientations = self.compute_keypoint_orientations(
                &keypoint, 
                magnitude_layer,
                orientation_layer
            );

            if orientations.len() == 1 {
                let mut updated_kp = keypoint;
                updated_kp.orientation = orientations[0];
                all_keypoints.push(updated_kp);
            } else {
                // 对于每个方向，创建一个新的关键点
                for &ori in &orientations {
                    let mut updated_kp = keypoint.clone();
                    updated_kp.orientation = ori;
                    all_keypoints.push(updated_kp);
                }
            }
        }

        // 然后计算描述符
        for keypoint in all_keypoints.iter_mut() {
            // 检查关键点的octave和scale是否在有效范围内
            if keypoint.octave >= gradient_magnitudes_pyramid.len() {
                continue;
            }
            
            let octave_data = &gradient_magnitudes_pyramid[keypoint.octave];
            let octave_orientation = &gradient_orientations_pyramid[keypoint.octave];
            
            if keypoint.scale as usize >= octave_data.len_of(Axis(0)) {
                continue;
            }
            
            let scale_idx = keypoint.scale as usize;
            let magnitude_layer = octave_data.index_axis(Axis(0), scale_idx);
            let orientation_layer = octave_orientation.index_axis(Axis(0), scale_idx);
            
            keypoint.descriptor = self.compute_descriptor(
                keypoint,
                magnitude_layer,
                orientation_layer
            );
        }
        
        *keypoints = all_keypoints;
    }

    fn compute_keypoint_orientations(
        &self,
        kp: &KeyPoint,
        magnitudes_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        orientations_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
    ) -> Vec<f32> {
        let (width, height) = (magnitudes_pyramid.shape()[1], magnitudes_pyramid.shape()[0]); // width是第1轴，height是第0轴
        
        // 计算高斯权重窗口的半径
        let sigma = self.gaussian_sigma_factor * kp.scale;
        let radius = (3.0 * sigma).round() as i32;
        
        let mut histogram = vec![0.0; self.num_bins];
        
        // 在关键点周围的区域内构建方向直方图
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let sample_x = (kp.x + dx as f32).round() as i32;
                let sample_y = (kp.y + dy as f32).round() as i32;
                
                // 检查边界
                if sample_x < 0 || sample_x >= width as i32 || sample_y < 0 || sample_y >= height as i32 {
                    continue;
                }

                let x = sample_x as usize;
                let y = sample_y as usize;

                let magnitude = magnitudes_pyramid[[y, x]];
                let orientation = orientations_pyramid[[y, x]];
                
                // 计算高斯权重
                let weight = (-(dx * dx + dy * dy) as f32 / (2.0 * sigma * sigma)).exp();
                let weighted_magnitude = magnitude * weight;
                
                // 将 [0, 2π] 映射到 [0, num_bins)
                let angle_bin = orientation * self.num_bins as f32 / (2.0 * std::f32::consts::PI);
                let bin_f = angle_bin;
                let bin = bin_f as usize % self.num_bins;
                
                // 线性插值到相邻的bin
                let fraction = bin_f - bin as f32;
                histogram[bin] += weighted_magnitude * (1.0 - fraction);
                histogram[(bin + 1) % self.num_bins] += weighted_magnitude * fraction;
            }
        }
        
        // 平滑直方图
        let smoothed_histogram = self.smooth_histogram(&histogram);
        
        // 找到所有超过阈值的峰值方向
        let max_value = smoothed_histogram.iter().cloned().fold(0.0, f32::max);
        let threshold = max_value * self.peak_ratio;
        
        let mut peaks = Vec::new();
        for i in 0..self.num_bins {
            let prev = smoothed_histogram[(i + self.num_bins - 1) % self.num_bins];
            let curr = smoothed_histogram[i];
            let next = smoothed_histogram[(i + 1) % self.num_bins];
            
            // 检查是否是局部最大值且超过阈值
            if curr > prev && curr > next && curr >= threshold {
                // 抛物线插值精确方向
                let interpolated_bin = i as f32 + 0.5 * (prev - next) / (prev - 2.0 * curr + next);
                let angle_rad = interpolated_bin * 2.0 * std::f32::consts::PI / self.num_bins as f32;
                peaks.push(angle_rad);
            }
        }
        
        if peaks.is_empty() {
            peaks.push(0.);
        }
        peaks
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

    fn compute_descriptor(
        &self,
        kp: &KeyPoint,
        magnitudes_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
        orientations_pyramid: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<[usize; 2]>>,
    ) -> Vec<f32> {
        let (width, height) = (magnitudes_pyramid.shape()[1], magnitudes_pyramid.shape()[0]); // width是第1轴，height是第0轴
        
        // 描述符向量初始化
        let mut descriptor_vector = vec![0.0; self.descriptor_width * self.descriptor_width * self.descriptor_num_bins];
        
        // 计算16x16邻域的大小参数
        let descriptor_size = 16.0;
        let cell_size = descriptor_size / self.descriptor_width as f32; // 4.0
        
        // 旋转矩阵（用于方向不变性）
        let cos_theta = kp.orientation.cos();
        let sin_theta = kp.orientation.sin();
        
        // 高斯权重参数（σ = 窗口宽度的一半）
        let sigma = 0.5 * descriptor_size;
        let invsig2 = 1.0 / (2.0 * sigma * sigma);
        
        // 在16x16窗口内采样
        for dy in -8..8 {
            for dx in -8..8 {
                // 计算相对于关键点的坐标
                let rx = dx as f32;
                let ry = dy as f32;
                
                // 旋转到关键点方向（相对于主方向）
                let tr_x = cos_theta * rx - sin_theta * ry;
                let tr_y = sin_theta * rx + cos_theta * ry;
                
                // 计算在原图上的坐标
                let x = kp.x + tr_x;
                let y = kp.y + tr_y;
                
                // 检查边界
                if x < 1.0 || y < 1.0 || x >= (width - 1) as f32 || y >= (height - 1) as f32 {
                    continue;
                }

                // 高斯窗口权重
                let win_weight = ((dx * dx + dy * dy) as f32 * -invsig2).exp();

                // 获取梯度幅度和方向
                let xi = x.floor() as usize;
                let yi = y.floor() as usize;
                
                // 确保索引有效
                if xi >= width - 1 || yi >= height - 1 {
                    continue;
                }
                
                let xf = x - xi as f32;
                let yf = y - yi as f32;
                
                // 双线性插值梯度幅度
                let mag_interp = magnitudes_pyramid[[yi, xi]] * (1.0 - xf) * (1.0 - yf) +
                                magnitudes_pyramid[[yi, xi + 1]] * xf * (1.0 - yf) +
                                magnitudes_pyramid[[yi + 1, xi]] * (1.0 - xf) * yf +
                                magnitudes_pyramid[[yi + 1, xi + 1]] * xf * yf;
                
                // 双线性插值梯度方向
                let ori_interp = orientations_pyramid[[yi, xi]] * (1.0 - xf) * (1.0 - yf) +
                                orientations_pyramid[[yi, xi + 1]] * xf * (1.0 - yf) +
                                orientations_pyramid[[yi + 1, xi]] * (1.0 - xf) * yf +
                                orientations_pyramid[[yi + 1, xi + 1]] * xf * yf;

                // 需要将梯度方向转换到关键点的局部坐标系中
                let mut angle_diff = ori_interp - kp.orientation;
                // 将角度差标准化到 [0, 2π)
                if angle_diff < 0.0 {
                    angle_diff += 2.0 * std::f32::consts::PI;
                }
                
                // 将梯度方向映射到bin
                let ori_bin = angle_diff * self.descriptor_num_bins as f32 / (2.0 * std::f32::consts::PI);
                
                // 计算在cell中的坐标
                let cx = (tr_x + descriptor_size / 2.0) / cell_size;
                let cy = (tr_y + descriptor_size / 2.0) / cell_size;
                
                if cx < -1.0 || cx >= self.descriptor_width as f32 || 
                   cy < -1.0 || cy >= self.descriptor_width as f32 {
                    continue;
                }

                // 三线性插值
                let c0 = (cx - 0.5).max(-1.0).min(self.descriptor_width as f32) as i32;
                let r0 = (cy - 0.5).max(-1.0).min(self.descriptor_width as f32) as i32;
                let o0 = ori_bin as i32;
                
                let dc = (cx - 0.5) - c0 as f32;
                let dr = (cy - 0.5) - r0 as f32;
                let do_ = ori_bin - o0 as f32;
                
                // 分配到相邻的bins
                for (rr, wr) in [(r0, 1.0 - dr), (r0 + 1, dr)] {
                    if rr < 0 || rr >= self.descriptor_width as i32 { continue; }
                    for (cc, wc) in [(c0, 1.0 - dc), (c0 + 1, dc)] {
                        if cc < 0 || cc >= self.descriptor_width as i32 { continue; }
                        for (oo, wo) in [(o0, 1.0 - do_), ((o0 + 1) % self.descriptor_num_bins as i32, do_)] {
                            let rridx = rr as usize;
                            let ccidx = cc as usize;
                            let ooidx = (oo + self.descriptor_num_bins as i32) % self.descriptor_num_bins as i32;
                            
                            if rridx < self.descriptor_width && ccidx < self.descriptor_width && ooidx >= 0 && ooidx < self.descriptor_num_bins as i32 {
                                let idx = (rridx * self.descriptor_width + ccidx) * self.descriptor_num_bins + ooidx as usize;
                                if idx < descriptor_vector.len() {
                                    let contrib = mag_interp * win_weight * wr * wc * wo;
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