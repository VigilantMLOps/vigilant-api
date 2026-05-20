CREATE TABLE IF NOT EXISTS reports (
    report_id     VARCHAR PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    report_type   VARCHAR NOT NULL,
    model_version VARCHAR,
    metrics       JSON,
    artifacts     JSON
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id   VARCHAR PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    severity      VARCHAR NOT NULL,
    incident_type VARCHAR NOT NULL,
    description   TEXT,
    status        VARCHAR NOT NULL DEFAULT 'TRIGGERED'
);

CREATE TABLE IF NOT EXISTS production_log (
    log_id      VARCHAR PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    features    JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id  VARCHAR PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level     VARCHAR NOT NULL,
    message   TEXT NOT NULL,
    metadata  JSON
);

CREATE TABLE IF NOT EXISTS feature_stats (
    feature_name VARCHAR PRIMARY KEY,
    stats_json   JSON NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
