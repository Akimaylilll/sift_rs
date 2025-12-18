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
                        magnitudes_pyramid[[scale, x, y]] = (dx * dx + dy * dy).sqrt();
                        orientations_pyramid[[scale, x, y]]= dy.atan2(dx); // atan2(dy, dx) 得到 [-π, π]
                    }
                }
            }
            gradient_magnitudes_pyramid.push(magnitudes_pyramid);
            gradient_orientations_pyramid.push(orientations_pyramid);
        }

        for keypoint in keypoints.iter_mut() {
        // println!("Keypoint at {}, {}, {}, {}, {}", keypoint.x, keypoint.y, keypoint.scale, keypoint.octave, keypoint.orientation);
            keypoint.orientation = self.compute_keypoint_orientations(&keypoint, 
              gradient_magnitudes_pyramid[keypoint.octave].index_axis(Axis(0), keypoint.scale as usize),
              gradient_orientations_pyramid[keypoint.octave].index_axis(Axis(0), keypoint.scale as usize)
            );
            // println!("Keypoint at {}, {}, {}, {}, {}", keypoint.x, keypoint.y, keypoint.scale, keypoint.octave, keypoint.orientation);
        }

        for keypoint in keypoints.iter_mut() {
            keypoint.descriptor = self.compute_descriptor(keypoint,
                gradient_magnitudes_pyramid[keypoint.octave].index_axis(Axis(0), keypoint.scale as usize),
                gradient_orientations_pyramid[keypoint.octave].index_axis(Axis(0), keypoint.scale as usize));
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
        let radius = 8;//(3.0 * gaussian_sigma_factor * kp.scale).round() as i32;
        let sigma = self.gaussian_sigma_factor * kp.scale;
        
        let mut histogram = vec![0.0; self.num_bins];
        
        // 在关键点周围的区域内构建方向直方图
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let region_x = kp.x as i32 + dx;
                let region_y = kp.y as i32 + dy;
                
                // 检查边界
                if region_x < 1 || region_x >= width as i32 - 1 || region_y < 1 || region_y >= height as i32 - 1 {
                    continue;
                }

                let gradient_magnitude = magnitudes_pyramid[[region_x as usize, region_y as usize]];
                let mut gradient_orientation = orientations_pyramid[[region_x as usize, region_y as usize]]; // atan2(dy, dx) 得到 [-π, π]
                
                // 确保方向在 [0, 2π) 范围内
                if gradient_orientation < 0.0 {
                    gradient_orientation += 2.0 * std::f32::consts::PI;
                }
                
                // 计算高斯权重
                let weight = (-(dx * dx + dy * dy) as f32 / (2.0 * sigma * sigma)).exp();
                let weighted_magnitude = gradient_magnitude * weight;
                
                // 找到对应的bin
                let bin_f = gradient_orientation * self.num_bins as f32 / (2.0 * std::f32::consts::PI);
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
        // 描述符区域大小（相对于关键点尺度）
        let descriptor_size = 3.0 * kp.scale;
        
        // 旋转矩阵（用于方向不变性）
        let cos_theta = kp.orientation.cos();
        let sin_theta = kp.orientation.sin();
        
        let mut descriptor_vector = vec![0.0; self.descriptor_width * self.descriptor_width * self.descriptor_num_bins];
        
        // 在描述符区域内采样
        let sample_step = descriptor_size / self.descriptor_width as f32;
        let sample_radius = (sample_step * 0.5) as i32;
        
        for sample_y in 0..self.descriptor_width {
            for sample_x in 0..self.descriptor_width {
                // 计算相对于关键点中心的采样位置
                let rel_x = (sample_x as f32 - self.descriptor_width as f32 * 0.5 + 0.5) * sample_step;
                let rel_y = (sample_y as f32 - self.descriptor_width as f32 * 0.5 + 0.5) * sample_step;
                
                // 旋转到关键点方向
                let rotated_x = rel_x * cos_theta - rel_y * sin_theta;
                let rotated_y = rel_x * sin_theta + rel_y * cos_theta;
                
                // 在采样点周围的小区域内累积梯度
                for dy in -sample_radius..=sample_radius {
                    for dx in -sample_radius..=sample_radius {
                        // 计算实际图像坐标
                        let sample_rel_x = rotated_x + dx as f32;
                        let sample_rel_y = rotated_y + dy as f32;
                        
                        // 旋转到关键点方向（再次确保方向正确）
                        let final_x = sample_rel_x * cos_theta - sample_rel_y * sin_theta;
                        let final_y = sample_rel_x * sin_theta + sample_rel_y * cos_theta;
                        
                        let image_x = kp.x + final_x;
                        let image_y = kp.y + final_y;
                        
                        // 检查边界
                        if image_x < 1.0 || image_x >= width as f32 - 1.0 ||
                           image_y < 1.0 || image_y >= height as f32 - 1.0 {
                            continue;
                        }
                        
                        let x = image_x as usize;
                        let y = image_y as usize;
                        
                        let magnitude = magnitudes_pyramid[[x, y]];
                        let mut orientation = orientations_pyramid[[x, y]];

                        // 归一化方向到 [0, 2π)
                        while orientation < 0.0 {
                            orientation += 2.0 * std::f32::consts::PI;
                        }
                        while orientation >= 2.0 * std::f32::consts::PI {
                            orientation -= 2.0 * std::f32::consts::PI;
                        }
                        
                        // 计算高斯权重（中心在采样点）
                        let gaussian_weight = (-(sample_rel_x * sample_rel_x + sample_rel_y * sample_rel_y) 
                            / (2.0 * (descriptor_size * 0.5).powi(2))).exp();
                        
                        let weighted_magnitude = magnitude * gaussian_weight;
                        
                        // 三线性插值
                        self.trilinear_interpolation(
                            sample_x, sample_y,
                            sample_rel_x, sample_rel_y,
                            orientation,
                            weighted_magnitude,
                            &mut descriptor_vector,
                        );
                    }
                }
            }
        }
        
        // 归一化描述符
        self.normalize_descriptor(&mut descriptor_vector);
        
        descriptor_vector
    }

    fn trilinear_interpolation(
        &self,
        bin_x: usize,
        bin_y: usize,
        rel_x: f32,
        rel_y: f32,
        orientation: f32,
        magnitude: f32,
        descriptor: &mut [f32],
    ) {
        // 计算相对于当前bin中心的偏移
        let bin_center_x = 0.0;
        let bin_center_y = 0.0;
        let x_offset = rel_x - bin_center_x;
        let y_offset = rel_y - bin_center_y;
        
        // 计算方向bin
        let orientation_bin = orientation * self.descriptor_num_bins as f32 / (2.0 * std::f32::consts::PI);
        let orientation_bin_floor = orientation_bin.floor() as i32;
        let orientation_frac = orientation_bin - orientation_bin_floor as f32;
        
        // 空间位置的权重
        let weight_x1 = (1.0 - x_offset.abs()).max(0.0);
        let weight_y1 = (1.0 - y_offset.abs()).max(0.0);
        
        for d_bin_y in 0..2 {
            for d_bin_x in 0..2 {
                let target_bin_x = (bin_x as i32 + d_bin_x) as usize;
                let target_bin_y = (bin_y as i32 + d_bin_y) as usize;
                
                if target_bin_x >= self.descriptor_width || target_bin_y >= self.descriptor_width {
                    continue;
                }
                
                // 空间权重
                let spatial_weight = if d_bin_x == 0 { weight_x1 } else { 1.0 - weight_x1 } *
                                    if d_bin_y == 0 { weight_y1 } else { 1.0 - weight_y1 };
                
                for d_orientation in 0..2 {
                    let target_orientation_bin = (orientation_bin_floor + d_orientation) as i32;
                    let wrapped_orientation_bin = (target_orientation_bin + self.descriptor_num_bins as i32) 
                        % self.descriptor_num_bins as i32;
                    
                    let orientation_weight = if d_orientation == 0 { 
                        1.0 - orientation_frac 
                    } else { 
                        orientation_frac 
                    };
                    
                    let total_weight = spatial_weight * orientation_weight * magnitude;
                    
                    let descriptor_idx = (target_bin_y * self.descriptor_width + target_bin_x) 
                        * self.descriptor_num_bins + wrapped_orientation_bin as usize;
                    
                    if descriptor_idx < descriptor.len() {
                        descriptor[descriptor_idx] += total_weight;
                    }
                }
            }
        }
      }
    
    fn normalize_descriptor(&self, descriptor: &mut [f32]) {
        let max_value = 0.2;
        // 计算L2范数
        let norm: f32 = descriptor.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm < std::f32::EPSILON {
            return;
        }
        
        // 第一次归一化
        for value in descriptor.iter_mut() {
            *value /= norm;
        }
        
        // 截断大值（增强对光照变化的鲁棒性）
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
        
        // 转换为u8以节省空间（可选）
        // 在实际应用中，通常存储为u8数组
        for value in descriptor.iter_mut() {
            *value = (*value * 512.0).min(255.0);
        }
    }
}