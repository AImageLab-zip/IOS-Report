from pydantic import BaseModel
from enum import Enum
import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
model = outlines.from_transformers(
    Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, device_map="auto",trust_remote_code=True),
    AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
)

class Rating(Enum):
    poor = 1
    fair = 2
    good = 3
    excellent = 4

class ProductReview(BaseModel):
    rating: Rating
    pros: list[str]
    cons: list[str]
    summary: str

review = model(
    "Review: The XPS 13 has great battery life and a stunning display, but it runs hot and the webcam is poor quality.",
    ProductReview,
    max_new_tokens=200,
)

review = ProductReview.model_validate_json(review)
print(f"Rating: {review.rating.name}")
print(f"Pros: {review.pros}")
print(f"Summary: {review.summary}")
print(review)