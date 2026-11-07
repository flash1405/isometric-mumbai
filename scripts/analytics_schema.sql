CREATE TABLE IF NOT EXISTS ev (ts INTEGER, host TEXT, sid TEXT, e TEXT, v TEXT, z REAL, country TEXT, city TEXT, ref TEXT, ua TEXT);
CREATE INDEX IF NOT EXISTS ev_e_ts ON ev (e, ts);
