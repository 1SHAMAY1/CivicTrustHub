# --------------------------------------------------------------
# CivicTrust Hub – Offline Tkinter UI (no external dependencies)
# --------------------------------------------------------------
# This self‑contained Python script implements the merged idea (Option 2 + 3)
# using only the Python standard library. It provides:
#   • Procurement anomaly detection (price outliers & contract‑splitting)
#   • Rumor verification via simple keyword overlap (no LLM needed)
#   • Auto‑generated RTI request & bilingual counter‑cards
#   • A downloadable ZIP packet containing the outputs and raw data
# --------------------------------------------------------------

import json
import csv
import math
import os
import zipfile
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Dict

# ==================== Data paths ====================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TENDERS_CSV = os.path.join(BASE_DIR, "tenders.csv")
RUMORS_JSON = os.path.join(BASE_DIR, "rumors.json")

# ------------------- Synthetic data creation -------------------
if not os.path.exists(TENDERS_CSV):
    rows = [
        ["T001", "Alpha Builders", 120000, "Road repair in Ward 5"],
        ["T002", "Beta Supplies", 50000, "Office stationery"],
        ["T003", "Gamma Corp", 75000, "School furniture"],
        ["T004", "Alpha Builders", 125000, "Road repair in Ward 5 (second contract)"],
        ["T005", "Delta Works", 30000, "Community hall painting"],
        ["T006", "Beta Supplies", 52000, "Office stationery – second batch"],
        ["T007", "Epsilon Ltd", 90000, "Water tank installation"],
        ["T008", "Gamma Corp", 77000, "School furniture – second batch"],
        ["T009", "Delta Works", 31000, "Community hall painting – second batch"],
        ["T010", "Zeta Services", 40000, "IT hardware purchase"],
    ]
    with open(TENDERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "vendor", "amount_usd", "description"])
        writer.writerows(rows)

if not os.path.exists(RUMORS_JSON):
    sample = [
        {"id": "R1", "text": "The city will shut down all schools tomorrow due to a flood.", "lang": "en"},
        {"id": "R2", "text": "डॉक्टरों का कहना है कि इस मौसम में एचआईवी का खतरना बढ़ेगा।", "lang": "hi"},
        {"id": "R3", "text": "No power cuts today, the grid is stable.", "lang": "en"},
        {"id": "R4", "text": "सरकार ने सभी सार्वजनिक पार्कों को 24 घंटे खोलने का आदेश दिया।", "lang": "hi"},
    ]
    with open(RUMORS_JSON, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

# ==================== Helper Functions ====================

def load_tenders() -> List[Dict]:
    with open(TENDERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def load_rumors() -> List[Dict]:
    with open(RUMORS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Procurement anomaly detection ----------
def detect_price_outliers(tenders: List[Dict]) -> List[Dict]:
    # Compute median per vendor and overall std
    amounts = [float(t["amount_usd"]) for t in tenders]
    overall_std = max(math.stdev(amounts) if len(amounts) > 1 else 1.0, 1.0)
    # group by vendor
    vendor_groups: Dict[str, List[float]] = {}
    for t in tenders:
        vendor_groups.setdefault(t["vendor"], []).append(float(t["amount_usd"]))
    vendor_medians = {v: (sorted(vals)[len(vals)//2]) for v, vals in vendor_groups.items()}
    for t in tenders:
        median = vendor_medians[t["vendor"]]
        z = (float(t["amount_usd"]) - median) / overall_std
        t["price_outlier"] = abs(z) > 2
    return tenders

def detect_contract_splitting(tenders: List[Dict]) -> List[Dict]:
    vendor_counts = {v: 0 for v in {t["vendor"] for t in tenders}}
    for t in tenders:
        vendor_counts[t["vendor"]] += 1
    for t in tenders:
        t["split_flag"] = vendor_counts[t["vendor"]] > 2
    return tenders

# ---------- Rumor verification (keyword fallback) ----------
# Very simple: if any word from official bulletins appears in rumor -> Verified
OFFICIALS = [
    "schools will remain open tomorrow",
    "no flood threat",
    "HIV transmission risk",
    "grid is stable",
    "public parks will stay closed",
]

def simple_verify(rumor: str) -> str:
    lowered = rumor.lower()
    for stmt in OFFICIALS:
        if all(word in lowered for word in stmt.split()):
            return "Verified"
    return "Unverified"

def verify_rumors(rumors: List[Dict]) -> List[Dict]:
    for r in rumors:
        r["verdict"] = simple_verify(r["text"])
    return rumors

# ==================== GUI ====================
class CivicTrustApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CivicTrust Hub – Procurement & Rumor Dashboard")
        self.geometry("900x600")
        self.resizable(True, True)
        self.tenders = detect_price_outliers(load_tenders())
        self.tenders = detect_contract_splitting(self.tenders)
        self.rumors = verify_rumors(load_rumors())
        self.create_widgets()

    def create_widgets(self):
        # ----- KPI frame -----
        kpi_frame = ttk.LabelFrame(self, text="Key Performance Indicators")
        kpi_frame.pack(fill="x", padx=10, pady=5)
        # Procurement KPIs
        outliers = sum(1 for t in self.tenders if t["price_outlier"])
        splits = sum(1 for t in self.tenders if t["split_flag"])
        panic = sum(1 for r in self.rumors if r["verdict"] == "Unverified")
        trust = sum(1 for r in self.rumors if r["verdict"] == "Verified")
        ttk.Label(kpi_frame, text=f"⚠️ Price Outliers: {outliers}").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(kpi_frame, text=f"🔀 Contract Splits: {splits}").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(kpi_frame, text=f"📈 Panic Index (unverified rumors): {panic}").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(kpi_frame, text=f"✅ Trust Index (verified rumors): {trust}").grid(row=0, column=3, padx=5, pady=2)

        # ----- Notebook tabs -----
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both", padx=10, pady=5)

        # Procurement tab
        proc_frame = ttk.Frame(notebook)
        notebook.add(proc_frame, text="Procurement Analyzer")
        self.build_procurement_tab(proc_frame)

        # Rumor tab
        rumor_frame = ttk.Frame(notebook)
        notebook.add(rumor_frame, text="Rumor Verifier")
        self.build_rumor_tab(rumor_frame)

        # Action packet download button
        dl_btn = ttk.Button(self, text="📦 Download Action Packet (ZIP)", command=self.download_packet)
        dl_btn.pack(pady=8)

    def build_procurement_tab(self, parent):
        flagged = [t for t in self.tenders if t["price_outlier"] or t["split_flag"]]
        if not flagged:
            ttk.Label(parent, text="✅ No anomalies detected.", foreground="green").pack(pady=10)
            return
        ttk.Label(parent, text=f"🚩 {len(flagged)} anomalous tenders:", foreground="orange").pack(pady=5)
        tree = ttk.Treeview(parent, columns=("id", "vendor", "amount", "price", "split", "desc"), show="headings")
        tree.heading("id", text="Tender ID")
        tree.heading("vendor", text="Vendor")
        tree.heading("amount", text="Amount (USD)")
        tree.heading("price", text="Price Outlier")
        tree.heading("split", text="Contract Split")
        tree.heading("desc", text="Description")
        for t in flagged:
            tree.insert("", "end", values=(t["id"], t["vendor"], t["amount_usd"], t["price_outlier"], t["split_flag"], t["description"]))
        tree.pack(expand=True, fill="both", padx=5, pady=5)

        # RTI request box
        ttk.Label(parent, text="Auto‑generated RTI Request:").pack(pady=5)
        rti_text = self.generate_rti(flagged)
        rti_box = tk.Text(parent, height=12, wrap="word")
        rti_box.insert("1.0", rti_text)
        rti_box.config(state="disabled")
        rti_box.pack(expand=False, fill="both", padx=5, pady=5)

    def generate_rti(self, flagged: List[Dict]) -> str:
        bullet_items = []
        for t in flagged:
            parts = []
            if t["price_outlier"]:
                parts.append("Price outlier")
            if t["split_flag"]:
                parts.append("Contract splitting")
            reason = " & ".join(parts) if parts else "Anomaly"
            bullet_items.append(f"- Tender ID **{t['id']}** (Vendor: {t['vendor']}, Amount: ${int(t['amount_usd'])}) – {reason}")
        items_str = "\n".join(bullet_items)
        template = (
            "To: Public Information Officer, Municipal Corporation\n\n"
            "Subject: Request for Information under Right to Information Act – Procurement Anomalies\n\n"
            "Dear Sir/Madam,\n\n"
            "Pursuant to Section 6 of the Right to Information Act, 2005, I hereby request copies of the following tender documents which appear anomalous based on publicly available data:\n\n"
            f"{items_str}\n\n"
            "Specifically, I seek:\n"
            "1. Detailed item‑wise cost breakdowns.\n"
            "2. Vendor selection criteria and evaluation scores.\n"
            "3. Any communications between the department and the listed vendors.\n\n"
            "Please provide the requested information within the statutory 30‑day period.\n\n"
            "Sincerely,\n[Your Name]\n[Contact Details]"
        )
        return template

    def build_rumor_tab(self, parent):
        for r in self.rumors:
            frame = ttk.Frame(parent, relief="ridge", borderwidth=1)
            frame.pack(fill="x", padx=5, pady=5)
            icon = "✅" if r["verdict"] == "Verified" else "⚠️"
            ttk.Label(frame, text=f"{icon} [{r['id']}] ({r['lang'].upper()})", font=("Helvetica", 10, "bold")).pack(anchor="w", padx=4, pady=2)
            ttk.Label(frame, text=r["text"], wraplength=800, justify="left").pack(anchor="w", padx=12)
            if r["verdict"] == "Unverified":
                en = "🚨 Official Update – The rumor above is unverified. No credible source confirms it at this time."
                hi = "🚨 आधिकारिक अपडेट – उपरोक्त अफवाह अविश्वसनीय है। वर्तमान में कोई विश्वसनीय स्रोत इसकी पुष्टि नहीं करता।"
                ttk.Label(frame, text=en + "\n" + hi, foreground="red", wraplength=800, justify="left").pack(anchor="w", padx=12, pady=4)

    def download_packet(self):
        # Build temporary zip in memory
        zip_path = os.path.join(BASE_DIR, "civictrust_packet.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # RTI (if any)
            flagged = [t for t in self.tenders if t["price_outlier"] or t["split_flag"]]
            if flagged:
                rti = self.generate_rti(flagged)
                zf.writestr("RTI_Request.txt", rti)
            # Counter‑cards for unverified rumors
            unverified = [r for r in self.rumors if r["verdict"] == "Unverified"]
            for r in unverified:
                en = "🚨 Official Update – The rumor above is unverified. No credible source confirms it at this time."
                hi = "🚨 आधिकारिक अपडेट – उपरोक्त अफवाह अविश्वसनीय है। वर्तमान में कोई विश्वसनीय स्रोत इसकी पुष्टि नहीं करता।"
                card = f"Rumor ID: {r['id']}\n\n" + en + "\n\n" + hi
                zf.writestr(f"counter_card_{r['id']}.txt", card)
            # Raw data
            zf.write(TENDERS_CSV, arcname="tenders.csv")
            zf.write(RUMORS_JSON, arcname="rumors.json")
        # Offer to save
        save_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP files", "*.zip")], initialfile="civictrust_action_packet.zip")
        if save_path:
            try:
                # copy temporary zip to chosen location
                with open(zip_path, "rb") as src, open(save_path, "wb") as dst:
                    dst.write(src.read())
                messagebox.showinfo("Success", f"Action packet saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save zip: {e}")
        # Clean up temporary file
        try:
            os.remove(zip_path)
        except OSError:
            pass

if __name__ == "__main__":
    app = CivicTrustApp()
    app.mainloop()
