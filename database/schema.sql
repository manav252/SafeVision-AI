CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE user_role AS ENUM ('admin', 'safety_officer', 'operator', 'viewer');
CREATE TYPE alert_severity AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
CREATE TYPE alert_status AS ENUM ('OPEN', 'ACKNOWLEDGED', 'RESOLVED');

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role user_role NOT NULL DEFAULT 'viewer',
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cameras (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(120) NOT NULL,
    stream_url TEXT,
    zone_name VARCHAR(120) NOT NULL DEFAULT 'Pending Setup',
    status VARCHAR(40) NOT NULL DEFAULT 'Pending Setup',
    restricted_zone JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS safety_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE SET NULL,
    zone_name VARCHAR(120) NOT NULL DEFAULT 'Plant',
    event_type VARCHAR(80) NOT NULL,
    severity alert_severity NOT NULL DEFAULT 'LOW',
    message TEXT NOT NULL,
    risk_score INTEGER NOT NULL DEFAULT 0,
    worker_id VARCHAR(120),
    evidence_uri TEXT,
    metadata_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID NOT NULL REFERENCES safety_events(id) ON DELETE CASCADE,
    title VARCHAR(160) NOT NULL,
    severity alert_severity NOT NULL,
    status alert_status NOT NULL DEFAULT 'OPEN',
    assigned_to VARCHAR(255),
    response_notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS plant_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    zone_name VARCHAR(120) NOT NULL,
    methane_lel DOUBLE PRECISION NOT NULL DEFAULT 0,
    co_ppm DOUBLE PRECISION NOT NULL DEFAULT 0,
    h2s_ppm DOUBLE PRECISION NOT NULL DEFAULT 0,
    oxygen_percent DOUBLE PRECISION NOT NULL DEFAULT 20.9,
    permit_type VARCHAR(120),
    equipment_status VARCHAR(160),
    shift_status VARCHAR(160),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_safety_events_type ON safety_events(event_type);
CREATE INDEX IF NOT EXISTS idx_safety_events_created_at ON safety_events(created_at);
CREATE INDEX IF NOT EXISTS idx_plant_signals_zone ON plant_signals(zone_name);

