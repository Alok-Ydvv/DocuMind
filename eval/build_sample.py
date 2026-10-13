"""Generate a synthetic 100-product catalog PDF and one evaluation question per product.

All products are fictional, so the expected answers are known exactly and no model has seen them.
Run from the project root: python -m eval.build_sample
"""
import json
import random
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

EVAL_DIR = Path(__file__).resolve().parent
PDF_PATH = EVAL_DIR / "data" / "sample_catalog.pdf"
QUESTIONS_PATH = EVAL_DIR / "questions.json"

SEED = 42
PRODUCT_COUNT = 100
ADJECTIVES = ["Zephyr", "Nimbus", "Aurora", "Cobalt", "Ember", "Glacier", "Harbor", "Juniper", "Keystone", "Lumen", "Meridian", "Orbit"]
NOUNS = ["Kettle", "Toaster", "Blender", "Lamp", "Speaker", "Router", "Heater", "Vacuum", "Mixer", "Fan", "Charger", "Dehumidifier"]
COLOURS = ["Teal", "Ivory", "Graphite", "Crimson", "Olive", "Navy"]
WARRANTY_MONTHS = [6, 12, 18, 24, 36]

# Each product gets one question. The field rotates so each type is asked 25 times.
FIELDS = ["warranty", "weight", "price", "colour"]


def make_products(rng: random.Random) -> list[dict]:
    names = rng.sample([f"{a} {n}" for a in ADJECTIVES for n in NOUNS], PRODUCT_COUNT)
    return [
        {
            "sku": f"SKU-{1001 + i}",
            "name": name,
            "colour": rng.choice(COLOURS),
            "warranty_months": rng.choice(WARRANTY_MONTHS),
            "weight_g": rng.randrange(120, 4800, 10),
            "price_usd": rng.randint(19, 399),
        }
        for i, name in enumerate(names)
    ]


def find_pages(pdf_path: Path, skus: list[str]) -> dict[str, int]:
    # Page numbers come from the rendered PDF, not from a guess about how records wrap.
    from pypdf import PdfReader

    pages = {}
    for number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        text = page.extract_text() or ""
        for sku in skus:
            if sku not in pages and sku in text:
                pages[sku] = number
    missing = set(skus) - set(pages)
    if missing:
        raise RuntimeError(f"SKUs not found in PDF: {sorted(missing)[:5]}")
    return pages


def build_pdf(products: list[dict]) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph("Northwind Home Appliances: Product Catalog", styles["Title"]), Spacer(1, 12)]
    for product in products:
        story.append(
            Paragraph(
                f"<b>{product['name']}</b><br/>"
                f"SKU: {product['sku']}<br/>"
                f"Colour: {product['colour']}<br/>"
                f"Warranty: {product['warranty_months']} months<br/>"
                f"Weight: {product['weight_g']} g<br/>"
                f"Price: ${product['price_usd']}",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 14))
    SimpleDocTemplate(str(PDF_PATH), pagesize=A4).build(story)


def make_question(product: dict, field: str, page: int) -> dict:
    name = product["name"]
    if field == "warranty":
        question = f"What is the warranty period for the {name}?"
        answer = f"{product['warranty_months']} months"
    elif field == "weight":
        question = f"How much does the {name} weigh?"
        answer = f"{product['weight_g']} g"
    elif field == "price":
        question = f"What is the price of the {name}?"
        answer = f"${product['price_usd']}"
    else:
        question = f"What colour is the {name} available in?"
        answer = product["colour"]
    return {
        "question": question,
        "answer": answer,
        "field": field,
        "sku": product["sku"],
        "page": page,
    }


def main() -> None:
    rng = random.Random(SEED)
    products = make_products(rng)
    build_pdf(products)
    pages = find_pages(PDF_PATH, [p["sku"] for p in products])
    questions = [
        make_question(p, FIELDS[i % len(FIELDS)], pages[p["sku"]]) for i, p in enumerate(products)
    ]
    QUESTIONS_PATH.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    print(f"Wrote {PDF_PATH} ({len(products)} products) and {QUESTIONS_PATH} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
