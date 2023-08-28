use std::cmp;

/// Clips the given value to the range [lower, upper].
pub fn clip<T>(value: T, lower: T, upper: T) -> T
where
    T: std::cmp::Ord,
{
    cmp::max(cmp::min(value, upper), lower)
}
