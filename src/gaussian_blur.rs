use ndarray::{Array2, ArrayView2};

pub struct GaussianFilter;

impl GaussianFilter {
    /// 创建一个新的高斯滤波器
    pub fn new() -> Self {
        GaussianFilter
    }
    
    /// 计算一维高斯核
    fn compute_kernel(sigma: f64) -> (usize, Vec<f64>) {
        // 计算核大小，确保为奇数
        let radius = (3.0 * sigma).ceil() as usize;
        let kernel_size = 2 * radius + 1;
        
        // 生成一维高斯核
        let mut kernel_1d = Vec::with_capacity(kernel_size);
        let sigma_sq_2 = 2.0 * sigma * sigma;
        
        // 计算高斯权重
        for i in 0..kernel_size {
            let x = i as f64 - radius as f64;
            let weight = (-x * x / sigma_sq_2).exp();
            kernel_1d.push(weight);
        }
        
        // 归一化核
        let sum: f64 = kernel_1d.iter().sum();
        for weight in &mut kernel_1d {
            *weight /= sum;
        }
        
        (kernel_size, kernel_1d)
    }
    
    /// 使用ndarray视图的优化版本，减少内存分配
    pub fn apply_to_array_view(input: ArrayView2<f32>, sigma: f64) -> Array2<f32> {
        let (height, width) = input.dim();
        
        // 根据 sigma 计算核
        let (kernel_size, kernel_1d) = Self::compute_kernel(sigma);
        let radius = kernel_size / 2;
        
        // 水平卷积
        let mut temp = Array2::<f32>::zeros((height, width));
        
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                
                for k in 0..kernel_size {
                    let k_pos = k as isize - radius as isize;
                    let sample_x = x as isize + k_pos;
                    
                    // 边界处理 - 使用镜像边界
                    let clamped_x = if sample_x < 0 {
                        (-sample_x) as usize
                    } else if sample_x >= width as isize {
                        (2 * width as isize - sample_x - 2) as usize
                    } else {
                        sample_x as usize
                    };
                    
                    // 确保索引在边界内
                    let valid_x = clamped_x.min(width - 1);
                    sum += input[[y, valid_x]] * kernel_1d[k] as f32;
                }
                
                temp[[y, x]] = sum;
            }
        }
        
        // 垂直卷积
        let mut result = Array2::<f32>::zeros((height, width));
        
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                
                for k in 0..kernel_size {
                    let k_pos = k as isize - radius as isize;
                    let sample_y = y as isize + k_pos;
                    
                    // 边界处理 - 使用镜像边界
                    let clamped_y = if sample_y < 0 {
                        (-sample_y) as usize
                    } else if sample_y >= height as isize {
                        (2 * height as isize - sample_y - 2) as usize
                    } else {
                        sample_y as usize
                    };
                    
                    // 确保索引在边界内
                    let valid_y = clamped_y.min(height - 1);
                    sum += temp[[valid_y, x]] * kernel_1d[k] as f32;
                }
                
                result[[y, x]] = sum;
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_apply_to_array2() {
        let input = Array2::from_shape_fn((5, 5), |(y, x)| {
            if y == 2 && x == 2 { 255.0 } else { 0.0 }
        });
        
        let result = GaussianFilter::apply_to_array_view(input.view(), 0.8);
        
        assert_eq!(result.shape(), &[5, 5]);
        // 中心点经过高斯滤波后应该扩散
        assert!(result[[2, 2]] > 0.0);
    }
}