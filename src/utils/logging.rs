use env_logger::Env;
use std::io::Write;

/// Initializes logger with custom formatting. Defaults to showing all INFO messages and above.
pub fn setup_logger() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format(|buf, record| {
            let ts = buf.timestamp_millis();
            let level = buf.default_styled_level(record.level());

            let mut dimmed = buf.style();
            dimmed.set_dimmed(true);

            writeln!(
                buf,
                "{}{} {} {}:{}{} {}",
                dimmed.value("["),
                dimmed.value(ts),
                dimmed.value(level),
                dimmed.value(record.file().unwrap()),
                dimmed.value(record.line().unwrap()),
                dimmed.value("]"),
                record.args()
            )
        })
        .init();
}
