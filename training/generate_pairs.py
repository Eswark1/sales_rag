"""
generate_pairs.py – Build a SQL Q&A JSONL training dataset from the seeded DB.
Outputs: training/data/sql_pairs.jsonl

Each record: {"instruction": "<question>", "output": "<sql>"}

Run: python training/generate_pairs.py
"""

import json
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "sales.db")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(OUT_DIR, "sql_pairs.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = [
    # ── Revenue ────────────────────────────────────────────────────────────
    ("What is the total revenue from all delivered orders?",
     "SELECT ROUND(SUM(o.total_amount),2) AS total_revenue FROM orders o WHERE o.status='Delivered';"),

    ("Show total revenue broken down by region.",
     "SELECT r.name AS region, ROUND(SUM(o.total_amount),2) AS revenue "
     "FROM orders o JOIN regions r ON o.region_id=r.region_id "
     "GROUP BY r.name ORDER BY revenue DESC;"),

    ("What is the revenue for each product category?",
     "SELECT p.category, ROUND(SUM(o.total_amount),2) AS revenue "
     "FROM orders o JOIN products p ON o.product_id=p.product_id "
     "GROUP BY p.category ORDER BY revenue DESC;"),

    ("What was monthly revenue for the last 6 months?",
     "SELECT strftime('%Y-%m', o.order_date) AS month, ROUND(SUM(o.total_amount),2) AS revenue "
     "FROM orders o "
     "WHERE o.order_date >= date('now','-6 months') AND o.status != 'Cancelled' "
     "GROUP BY month ORDER BY month;"),

    ("Which sales rep generated the most revenue this year?",
     "SELECT s.full_name, ROUND(SUM(o.total_amount),2) AS revenue "
     "FROM orders o JOIN sales_reps s ON o.rep_id=s.rep_id "
     "WHERE strftime('%Y', o.order_date)=strftime('%Y','now') "
     "GROUP BY s.full_name ORDER BY revenue DESC LIMIT 1;"),

    # ── Pipeline ───────────────────────────────────────────────────────────
    ("How many opportunities are in each pipeline stage?",
     "SELECT stage, COUNT(*) AS count FROM opportunities GROUP BY stage ORDER BY count DESC;"),

    ("What is the total estimated pipeline value?",
     "SELECT ROUND(SUM(estimated_value),2) AS pipeline_value "
     "FROM opportunities WHERE stage NOT IN ('Closed Won','Closed Lost');"),

    ("What is the weighted pipeline value (value × probability)?",
     "SELECT ROUND(SUM(estimated_value * probability),2) AS weighted_pipeline "
     "FROM opportunities WHERE stage NOT IN ('Closed Won','Closed Lost');"),

    ("Show the win rate as a percentage.",
     "SELECT ROUND(100.0 * SUM(CASE WHEN stage='Closed Won' THEN 1 ELSE 0 END) "
     "/ COUNT(*), 1) AS win_rate_pct FROM opportunities "
     "WHERE stage IN ('Closed Won','Closed Lost');"),

    ("Which rep has the most open opportunities?",
     "SELECT s.full_name, COUNT(*) AS open_opps "
     "FROM opportunities o JOIN sales_reps s ON o.rep_id=s.rep_id "
     "WHERE o.stage NOT IN ('Closed Won','Closed Lost') "
     "GROUP BY s.full_name ORDER BY open_opps DESC LIMIT 5;"),

    # ── Leads ──────────────────────────────────────────────────────────────
    ("What is the lead conversion rate from qualified to won?",
     "SELECT ROUND(100.0 * won / qualified, 1) AS conversion_pct FROM ("
     "  SELECT "
     "    SUM(CASE WHEN l.status='Qualified' THEN 1 ELSE 0 END) AS qualified,"
     "    SUM(CASE WHEN o.stage='Closed Won' THEN 1 ELSE 0 END) AS won "
     "  FROM leads l LEFT JOIN opportunities o ON l.lead_id=o.lead_id"
     ");"),

    ("How many new leads came in this month by source?",
     "SELECT source, COUNT(*) AS leads "
     "FROM leads "
     "WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m','now') "
     "GROUP BY source ORDER BY leads DESC;"),

    ("Which region has the most disqualified leads?",
     "SELECT r.name AS region, COUNT(*) AS disqualified "
     "FROM leads l JOIN regions r ON l.region_id=r.region_id "
     "WHERE l.status='Disqualified' "
     "GROUP BY r.name ORDER BY disqualified DESC LIMIT 5;"),

    # ── Orders ─────────────────────────────────────────────────────────────
    ("What is the average order value by product?",
     "SELECT p.name AS product, ROUND(AVG(o.total_amount),2) AS avg_order_value "
     "FROM orders o JOIN products p ON o.product_id=p.product_id "
     "GROUP BY p.name ORDER BY avg_order_value DESC;"),

    ("How many orders are currently in Pending or Processing status?",
     "SELECT status, COUNT(*) AS count FROM orders "
     "WHERE status IN ('Pending','Processing') GROUP BY status;"),

    ("What is the average discount applied to orders?",
     "SELECT ROUND(AVG(discount_pct)*100, 1) AS avg_discount_pct FROM orders;"),

    ("Show me the top 10 customers by total spend.",
     "SELECT customer_name, ROUND(SUM(total_amount),2) AS total_spend "
     "FROM orders WHERE status='Delivered' "
     "GROUP BY customer_name ORDER BY total_spend DESC LIMIT 10;"),

    # ── Activities ─────────────────────────────────────────────────────────
    ("How many activities has each rep logged this month?",
     "SELECT s.full_name, COUNT(*) AS activities "
     "FROM activities a JOIN sales_reps s ON a.rep_id=s.rep_id "
     "WHERE strftime('%Y-%m', a.occurred_at) = strftime('%Y-%m','now') "
     "GROUP BY s.full_name ORDER BY activities DESC;"),

    ("What are the most common activity types across all deals?",
     "SELECT activity_type, COUNT(*) AS count FROM activities "
     "GROUP BY activity_type ORDER BY count DESC;"),
]


def main():
    # Verify queries run against the real DB
    conn = sqlite3.connect(DB_PATH)
    valid, skipped = 0, 0
    records = []
    for question, sql in PAIRS:
        try:
            conn.execute(sql)
            records.append({"instruction": question, "output": sql.strip()})
            valid += 1
        except sqlite3.Error as e:
            print(f"SKIP (error: {e}): {question[:60]}...")
            skipped += 1
    conn.close()

    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {valid} valid pairs -> {OUT_PATH}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
