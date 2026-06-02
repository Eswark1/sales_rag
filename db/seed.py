"""
seed.py – Populate the sales RAG SQLite database with realistic sample data.
Run:  python db/seed.py
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "sales.db"))
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

random.seed(42)


def random_date(start_days_ago: int, end_days_ago: int = 0) -> str:
    delta = random.randint(end_days_ago, start_days_ago)
    return (datetime.now() - timedelta(days=delta)).strftime("%Y-%m-%d")


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # Apply schema
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    # ── Regions ────────────────────────────────────────────
    regions = ["North America", "EMEA", "APAC", "LATAM", "ANZ"]
    cur.executemany("INSERT OR IGNORE INTO regions(name) VALUES (?)",
                    [(r,) for r in regions])
    conn.commit()
    cur.execute("SELECT region_id, name FROM regions")
    region_map = {name: rid for rid, name in cur.fetchall()}

    # ── Products ───────────────────────────────────────────
    products = [
        ("CRM Pro", "Software", 1200.00, "Full-featured CRM platform, per seat/year"),
        ("Analytics Suite", "Software", 2500.00, "BI and analytics dashboard bundle"),
        ("Data Connector Pack", "Add-on", 400.00, "50+ pre-built data connectors"),
        ("Support Gold", "Service", 800.00, "24/7 priority support contract"),
        ("Implementation Services", "Service", 5000.00, "Onboarding and configuration package"),
        ("Mobile SDK", "Software", 600.00, "iOS & Android SDK license"),
        ("AI Insights Module", "Add-on", 1800.00, "LLM-powered sales insights"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO products(name, category, unit_price, description) VALUES (?,?,?,?)",
        products,
    )
    conn.commit()
    cur.execute("SELECT product_id, name FROM products")
    product_map = {name: pid for pid, name in cur.fetchall()}
    product_ids = list(product_map.values())

    # ── Sales Reps ─────────────────────────────────────────
    reps = [
        ("Alice Martinez", "alice@acmecorp.io", "North America"),
        ("Brian Chen", "brian@acmecorp.io", "APAC"),
        ("Carol Smith", "carol@acmecorp.io", "EMEA"),
        ("David Okafor", "david@acmecorp.io", "LATAM"),
        ("Eva Petersen", "eva@acmecorp.io", "ANZ"),
        ("Frank Liu", "frank@acmecorp.io", "North America"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO sales_reps(full_name, email, region_id, hire_date) VALUES (?,?,?,?)",
        [(name, email, region_map[region], random_date(1200, 300))
         for name, email, region in reps],
    )
    conn.commit()
    cur.execute("SELECT rep_id, full_name FROM sales_reps")
    rep_map = {name: rid for rid, name in cur.fetchall()}
    rep_ids = list(rep_map.values())

    # ── Leads ──────────────────────────────────────────────
    companies = [
        "Horizon Retail", "Apex Logistics", "BlueSky Analytics", "NovaTech Corp",
        "Greenfield Energy", "Pinnacle Finance", "Coastline Media", "Summit Health",
        "Quantum Dynamics", "RedRock Manufacturing", "Starfield Consulting",
        "Lakeside Pharma", "Iron Bridge Capital", "Crestview Hotels", "Orion EdTech",
        "Meridian Bank", "Cascade Telecom", "Vortex Gaming", "Polaris Aerospace",
        "Delta Foods",
    ]
    sources = ["Website", "Referral", "Cold Call", "Event", "LinkedIn", "Other"]
    lead_statuses = ["New", "Contacted", "Qualified", "Disqualified"]
    lead_weights = [0.25, 0.30, 0.35, 0.10]
    first_names = ["James", "Sarah", "Tom", "Lisa", "Mike", "Anna", "John", "Kate"]
    last_names = ["Johnson", "Williams", "Brown", "Davis", "Wilson", "Taylor", "Moore"]

    lead_rows = []
    for company in companies:
        contact = f"{random.choice(first_names)} {random.choice(last_names)}"
        status = random.choices(lead_statuses, weights=lead_weights)[0]
        rep_id = random.choice(rep_ids)
        region_id = random.choice(list(region_map.values()))
        lead_rows.append((
            company, contact, f"{contact.lower().replace(' ', '.')}@{company.lower().replace(' ', '')}.com",
            f"+1-555-{random.randint(1000,9999)}", random.choice(sources),
            status, rep_id, region_id, random_date(365, 1),
            f"Initial outreach via {random.choice(sources).lower()}."
        ))

    cur.executemany(
        """INSERT INTO leads
           (company_name, contact_name, contact_email, contact_phone,
            source, status, rep_id, region_id, created_at, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        lead_rows,
    )
    conn.commit()
    cur.execute("SELECT lead_id FROM leads")
    lead_ids = [r[0] for r in cur.fetchall()]

    # ── Opportunities ──────────────────────────────────────
    stages = ["Prospecting","Qualification","Needs Analysis",
              "Proposal","Negotiation","Closed Won","Closed Lost"]
    stage_weights = [0.10, 0.15, 0.20, 0.20, 0.15, 0.15, 0.05]

    opp_rows = []
    for lead_id in lead_ids:
        # 1-3 opportunities per qualified lead
        cur.execute("SELECT status, rep_id, region_id FROM leads WHERE lead_id=?", (lead_id,))
        lead_status, rep_id, region_id = cur.fetchone()
        if lead_status == "Disqualified":
            continue
        n_opps = random.randint(1, 3)
        for _ in range(n_opps):
            pid = random.choice(product_ids)
            cur.execute("SELECT unit_price FROM products WHERE product_id=?", (pid,))
            base_price = cur.fetchone()[0]
            seats = random.randint(5, 200)
            value = round(base_price * seats * random.uniform(0.8, 1.2), 2)
            stage = random.choices(stages, weights=stage_weights)[0]
            prob = {"Prospecting":0.10,"Qualification":0.25,"Needs Analysis":0.40,
                    "Proposal":0.60,"Negotiation":0.75,"Closed Won":1.0,"Closed Lost":0.0}[stage]
            created = random_date(300, 30)
            expected_close = (datetime.strptime(created, "%Y-%m-%d") + timedelta(days=random.randint(30,180))).strftime("%Y-%m-%d")
            actual_close = expected_close if stage in ("Closed Won","Closed Lost") else None
            opp_rows.append((
                lead_id, rep_id,
                f"Opp – {random.choice(['Expansion','New Business','Renewal','Upsell'])}",
                stage, value, prob, expected_close, actual_close,
                pid, region_id, created, created,
                f"{stage} stage – {random.choice(['strong interest','price negotiation','awaiting sign-off','demo scheduled'])}."
            ))

    cur.executemany(
        """INSERT INTO opportunities
           (lead_id, rep_id, name, stage, estimated_value, probability,
            expected_close, actual_close, product_id, region_id,
            created_at, updated_at, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        opp_rows,
    )
    conn.commit()

    cur.execute("SELECT opportunity_id, rep_id, product_id, region_id FROM opportunities WHERE stage='Closed Won'")
    won_opps = cur.fetchall()

    # ── Orders ─────────────────────────────────────────────
    order_statuses = ["Pending","Processing","Shipped","Delivered","Cancelled"]
    order_weights  = [0.05, 0.10, 0.15, 0.65, 0.05]

    order_rows = []
    for opp_id, rep_id, pid, region_id in won_opps:
        cur.execute("SELECT unit_price FROM products WHERE product_id=?", (pid,))
        unit_price = cur.fetchone()[0]
        qty = random.randint(1, 50)
        discount = round(random.choice([0, 0.05, 0.10, 0.15, 0.20]), 2)
        status = random.choices(order_statuses, weights=order_weights)[0]
        order_date = random_date(180, 1)
        shipped = None
        if status in ("Shipped","Delivered"):
            shipped = (datetime.strptime(order_date, "%Y-%m-%d") + timedelta(days=random.randint(1,7))).strftime("%Y-%m-%d")
        # pull customer from opportunity's lead
        cur.execute("""SELECT l.company_name, l.contact_email
                       FROM leads l JOIN opportunities o ON l.lead_id=o.lead_id
                       WHERE o.opportunity_id=?""", (opp_id,))
        row = cur.fetchone()
        customer_name = row[0] if row else "Unknown"
        customer_email = row[1] if row else None

        order_rows.append((
            opp_id, rep_id, customer_name, customer_email,
            pid, qty, unit_price, discount,
            status, order_date, shipped, region_id,
            f"Order generated from won opportunity {opp_id}."
        ))

    cur.executemany(
        """INSERT INTO orders
           (opportunity_id, rep_id, customer_name, customer_email,
            product_id, quantity, unit_price, discount_pct,
            status, order_date, shipped_date, region_id, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        order_rows,
    )
    conn.commit()

    # ── Activities ─────────────────────────────────────────
    activity_types = ["Email","Call","Meeting","Demo","Proposal Sent","Follow-up","Contract Sent","Note"]
    activity_rows = []

    # Activities on leads
    cur.execute("SELECT lead_id, rep_id FROM leads")
    for lead_id, rep_id in cur.fetchall():
        for _ in range(random.randint(1, 4)):
            atype = random.choice(activity_types)
            activity_rows.append((
                "Lead", lead_id, rep_id, atype,
                f"{atype} with lead {lead_id}: {random.choice(['discussed pricing','confirmed budget','sent materials','scheduled demo','no response yet'])}.",
                random_date(300, 1)
            ))

    # Activities on opportunities
    cur.execute("SELECT opportunity_id, rep_id FROM opportunities")
    for opp_id, rep_id in cur.fetchall():
        for _ in range(random.randint(1, 5)):
            atype = random.choice(activity_types)
            activity_rows.append((
                "Opportunity", opp_id, rep_id, atype,
                f"{atype} on opportunity {opp_id}: {random.choice(['proposal reviewed','demo delivered','legal reviewing contract','stakeholder alignment call','awaiting approval'])}.",
                random_date(200, 1)
            ))

    cur.executemany(
        """INSERT INTO activities(entity_type, entity_id, rep_id, activity_type, summary, occurred_at)
           VALUES (?,?,?,?,?,?)""",
        activity_rows,
    )
    conn.commit()
    conn.close()

    print(f"Database seeded successfully -> {DB_PATH}")
    print(f"  {len(companies)} leads | {len(opp_rows)} opportunities | {len(order_rows)} orders | {len(activity_rows)} activities")


if __name__ == "__main__":
    main()
