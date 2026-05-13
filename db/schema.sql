-- ============================================================
-- Sales RAG Database Schema
-- ============================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------
-- Reference / Lookup tables
-- ----------------------------------------------------------

CREATE TABLE IF NOT EXISTS regions (
    region_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS products (
    product_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    unit_price  REAL NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS sales_reps (
    rep_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    region_id   INTEGER REFERENCES regions(region_id),
    hire_date   TEXT NOT NULL   -- ISO-8601
);

-- ----------------------------------------------------------
-- Core sales pipeline tables
-- ----------------------------------------------------------

CREATE TABLE IF NOT EXISTS leads (
    lead_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name    TEXT NOT NULL,
    contact_name    TEXT NOT NULL,
    contact_email   TEXT,
    contact_phone   TEXT,
    source          TEXT CHECK(source IN ('Website','Referral','Cold Call','Event','LinkedIn','Other')),
    status          TEXT NOT NULL DEFAULT 'New'
                        CHECK(status IN ('New','Contacted','Qualified','Disqualified')),
    rep_id          INTEGER REFERENCES sales_reps(rep_id),
    region_id       INTEGER REFERENCES regions(region_id),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS opportunities (
    opportunity_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id         INTEGER REFERENCES leads(lead_id),
    rep_id          INTEGER REFERENCES sales_reps(rep_id),
    name            TEXT NOT NULL,
    stage           TEXT NOT NULL DEFAULT 'Prospecting'
                        CHECK(stage IN (
                            'Prospecting','Qualification','Needs Analysis',
                            'Proposal','Negotiation','Closed Won','Closed Lost'
                        )),
    estimated_value REAL NOT NULL DEFAULT 0,
    probability     REAL NOT NULL DEFAULT 0 CHECK(probability BETWEEN 0 AND 1),
    expected_close  TEXT,           -- ISO-8601 date
    actual_close    TEXT,
    product_id      INTEGER REFERENCES products(product_id),
    region_id       INTEGER REFERENCES regions(region_id),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id  INTEGER REFERENCES opportunities(opportunity_id),
    rep_id          INTEGER REFERENCES sales_reps(rep_id),
    customer_name   TEXT NOT NULL,
    customer_email  TEXT,
    product_id      INTEGER REFERENCES products(product_id),
    quantity        INTEGER NOT NULL DEFAULT 1,
    unit_price      REAL NOT NULL,
    discount_pct    REAL NOT NULL DEFAULT 0 CHECK(discount_pct BETWEEN 0 AND 1),
    total_amount    REAL GENERATED ALWAYS AS
                        (ROUND(quantity * unit_price * (1 - discount_pct), 2)) STORED,
    status          TEXT NOT NULL DEFAULT 'Pending'
                        CHECK(status IN ('Pending','Processing','Shipped','Delivered','Cancelled','Refunded')),
    order_date      TEXT NOT NULL DEFAULT (datetime('now')),
    shipped_date    TEXT,
    region_id       INTEGER REFERENCES regions(region_id),
    notes           TEXT
);

-- ----------------------------------------------------------
-- Activity / interaction log (useful for RAG context)
-- ----------------------------------------------------------

CREATE TABLE IF NOT EXISTS activities (
    activity_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL CHECK(entity_type IN ('Lead','Opportunity','Order')),
    entity_id       INTEGER NOT NULL,
    rep_id          INTEGER REFERENCES sales_reps(rep_id),
    activity_type   TEXT NOT NULL CHECK(activity_type IN (
                        'Email','Call','Meeting','Demo','Proposal Sent',
                        'Follow-up','Contract Sent','Note')),
    summary         TEXT NOT NULL,
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ----------------------------------------------------------
-- Indexes for common query patterns
-- ----------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_leads_status        ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_rep           ON leads(rep_id);
CREATE INDEX IF NOT EXISTS idx_opp_stage           ON opportunities(stage);
CREATE INDEX IF NOT EXISTS idx_opp_rep             ON opportunities(rep_id);
CREATE INDEX IF NOT EXISTS idx_opp_lead            ON opportunities(lead_id);
CREATE INDEX IF NOT EXISTS idx_orders_status       ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_rep          ON orders(rep_id);
CREATE INDEX IF NOT EXISTS idx_activities_entity   ON activities(entity_type, entity_id);
