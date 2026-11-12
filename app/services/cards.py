"""Card freeze/unfreeze — deterministic code only. The LLM never writes to
the database: it may detect a "freeze my card" intent (see
app/services/intent.py) and the chat layer asks for explicit confirmation,
but the actual status change always happens here, only after a customer
clicks Yes, and only on a card verified to belong to that customer."""
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import ActionLog, Card


@dataclass
class ActionResult:
    ok: bool
    message: str
    status: str | None = None


def get_customer_card(db: Session, customer_id: str) -> Card | None:
    return db.query(Card).filter(Card.customer_id == customer_id).first()


def _log(db: Session, customer_id: str, action: str, status: str, details: str) -> None:
    db.add(ActionLog(customer_id=customer_id, action=action, status=status, details=details))
    db.commit()


def freeze_card(db: Session, customer_id: str, card_id: str) -> ActionResult:
    card = db.get(Card, card_id)
    if card is None or card.customer_id != customer_id:
        _log(db, customer_id, "freeze_card", "denied", f"card {card_id} not owned by customer")
        return ActionResult(ok=False, message="We couldn't find that card on your account.")

    if card.status == "frozen":
        _log(db, customer_id, "freeze_card", "no_op", f"card ending {card.last4} already frozen")
        return ActionResult(
            ok=True, message=f"Your card ending {card.last4} is already frozen.", status="frozen"
        )

    card.status = "frozen"
    db.add(card)
    _log(db, customer_id, "freeze_card", "done", f"card ending {card.last4} frozen")
    return ActionResult(ok=True, message=f"Your card ending {card.last4} has been frozen.", status="frozen")


def unfreeze_card(db: Session, customer_id: str, card_id: str) -> ActionResult:
    card = db.get(Card, card_id)
    if card is None or card.customer_id != customer_id:
        _log(db, customer_id, "unfreeze_card", "denied", f"card {card_id} not owned by customer")
        return ActionResult(ok=False, message="We couldn't find that card on your account.")

    if card.status == "active":
        _log(db, customer_id, "unfreeze_card", "no_op", f"card ending {card.last4} already active")
        return ActionResult(
            ok=True, message=f"Your card ending {card.last4} is already active.", status="active"
        )

    card.status = "active"
    db.add(card)
    _log(db, customer_id, "unfreeze_card", "done", f"card ending {card.last4} unfrozen")
    return ActionResult(ok=True, message=f"Your card ending {card.last4} has been unfrozen.", status="active")
