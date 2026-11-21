//! Calibrate on past forecasts, then check separate observed outcomes.
//! Each row holds base total, base left, base right, observed left, observed right.

use faer::Mat;
use sheaf::{HierarchicalConformal, ReconciliationMethod, SummingMatrix};
use std::{error::Error, fs, path::Path};

type Measurements = (Mat<f64>, Mat<f64>);

fn read_forecasts(path: &Path) -> Result<Measurements, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    parse_forecasts(path, &text)
}

fn parse_forecasts(path: &Path, text: &str) -> Result<Measurements, Box<dyn Error>> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<f64> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        let row: [f64; 5] = values
            .try_into()
            .map_err(|_| format!("{}:{}: expected five numbers", path.display(), i + 1))?;
        if !row.iter().all(|v| v.is_finite()) || !(row[3] + row[4]).is_finite() {
            return Err(format!("{}:{}: non-finite value or total", path.display(), i + 1).into());
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{}: no forecasts", path.display()).into());
    }
    let forecasts = Mat::from_fn(3, rows.len(), |i, j| rows[j][i]);
    let actual = Mat::from_fn(3, rows.len(), |i, j| match i {
        0 => rows[j][3] + rows[j][4],
        1 => rows[j][3],
        _ => rows[j][4],
    });
    Ok((forecasts, actual))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [calibration, test, alpha] = args.as_slice() else {
        return Err("usage: forecast_coverage CALIBRATION TEST ALPHA".into());
    };
    let alpha = alpha.to_str().ok_or("alpha must be a number")?.parse()?;
    let (base, actual) = read_forecasts(calibration.as_ref())?;
    let mut model =
        HierarchicalConformal::new(SummingMatrix::simple_star(2), ReconciliationMethod::Ols);
    model.calibrate(&actual, &base, alpha)?;

    let (base, actual) = read_forecasts(test.as_ref())?;
    let center = model.predict(&base)?;
    let (lower, upper) = model.predict_intervals(&base)?;
    let radius = model.quantile();
    println!("radius {radius:.4}");
    println!("row total left right residual in_ball in_box");
    for j in 0..base.ncols() {
        let residual = |i: usize| actual[(i, j)] - center[(i, j)];
        let norm = (0..3).map(|i| residual(i).powi(2)).sum::<f64>().sqrt();
        let in_box = (0..3).all(|i| (lower[(i, j)]..=upper[(i, j)]).contains(&actual[(i, j)]));
        println!(
            "{} {:.4} {:.4} {:.4} {norm:.4} {} {in_box}",
            j + 1,
            center[(0, j)],
            center[(1, j)],
            center[(2, j)],
            norm <= radius,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_forecasts_and_observations_sum_to_the_total() -> Result<(), Box<dyn Error>> {
        let (base, actual) =
            parse_forecasts(Path::new("data"), "33 10 20 11 21\n\n30 10 20 10 20")?;
        assert_eq!((base.nrows(), base.ncols()), (3, 2));
        assert_eq!(base[(0, 0)], 33.0);
        assert_eq!(actual[(0, 0)], 32.0);
        assert_eq!(actual[(1, 0)], 11.0);
        assert_eq!(actual[(2, 0)], 21.0);
        assert_eq!(actual[(0, 1)], 30.0);
        Ok(())
    }

    #[test]
    fn malformed_rows_and_overflowing_observations_are_rejected() {
        let path = Path::new("calibration");
        for row in [
            "",
            "1 2 3 4",
            "1 2 3 4 5 6",
            "x 2 3 4 5",
            "1 2 3 NaN 5",
            "1 2 3 1e308 1e308",
        ] {
            assert!(parse_forecasts(path, row).is_err(), "accepted {row:?}");
        }
    }
}
