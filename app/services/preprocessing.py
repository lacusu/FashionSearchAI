import re
import pandas as pd
from app.utils.settings import IMAGE_URL_PREFIX

def strip_html(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<.*?>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_attr(s: str):
    try:
        s = s.replace("None", "'NA'")
        return eval(s)
    except Exception:
        return {}

def enrich_tags(name: str, desc: str) -> str:
    t = f"{name} {desc}".lower()
    tags = []
    if any(k in t for k in ["party","festive","wedding","sequins","embellish"]):
        tags.append("occasion:party")
    if any(k in t for k in ["formal","office","work","slim fit"]):
        tags.append("occasion:formal")
    if any(k in t for k in ["cotton","linen","breathable","summer"]):
        tags.append("season:summer")
    if any(k in t for k in ["wool","sweater","hoodie","winter"]):
        tags.append("season:winter")
    if any(k in t for k in ["kurta","ethnic","dupatta","palazzos"]):
        tags.append("style:ethnic")
    return " ".join(tags)

def build_chunks(df: pd.DataFrame) -> pd.DataFrame:
    # Clean expected text fields (your CSV structure unchanged)
    for col in ["name","products","colour","brand","description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(strip_html)
        else:
            df[col] = ""

    # Attributes (optional column)
    if "p_attributes" in df.columns:
        df["attributes"] = df["p_attributes"].apply(parse_attr)
    else:
        df["attributes"] = {}

    # Derive image URL from p_id if present
    def image_url(row):
        if "p_id" in row and pd.notna(row["p_id"]):
            return f"{IMAGE_URL_PREFIX}/{str(row['p_id']).strip()}.jpg"
        return None

    df["image"] = df.apply(image_url, axis=1)

    def product_text(row):
        # flatten attributes if any
        attrs = row.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        attr_kv = " ".join([f"{k}:{v}" for k, v in attrs.items() if isinstance(v, str)])
        return " | ".join([
            f"name: {row.get('name','')}",
            f"brand: {row.get('brand','')}",
            f"colour: {row.get('colour','')}",
            f"price: {row.get('price','')}",
            f"desc: {row.get('description','')}",
            f"tags: {attr_kv}"
        ])

    df["chunk_product"] = df.apply(product_text, axis=1)
    df["tags_inferred"] = df.apply(lambda r: enrich_tags(r.get('name',''), r.get('description','')), axis=1)
    df["chunk_hybrid"] = df["chunk_product"] + " | " + df["tags_inferred"]

    return df
