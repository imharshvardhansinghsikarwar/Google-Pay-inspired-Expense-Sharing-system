import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any

class ExpenseSharingSystem:
    def __init__(self):
        # Transaction schema: transaction_id, date, description, payer, amount, split_type, category
        self.transactions_df = pd.DataFrame(columns=[
            "txn_id", "date", "description", "payer", "amount", "split_type", "category"
        ])
        # Participant shares: txn_id, user, share_amount
        self.splits_df = pd.DataFrame(columns=["txn_id", "user", "share_amount"])

    def add_expense(
        self, 
        txn_id: str, 
        date: str, 
        payer: str, 
        amount: float, 
        description: str, 
        category: str, 
        participants: List[str], 
        split_type: str = "equal", 
        shares: Dict[str, float] = None
    ):
        """
        Records an expense and generates participant-level debit records.
        """
        assert amount > 0, "Expense amount must be positive."
        assert payer in participants or split_type in ["equal", "weighted", "exact"], "Invalid participants."

        # Compute breakdown
        allocated_shares = {}
        if split_type == "equal":
            base_share = np.round(amount / len(participants), 2)
            allocated_shares = {user: base_share for user in participants}
            # Balance residual cents to payer to guarantee sum == amount
            diff = round(amount - sum(allocated_shares.values()), 2)
            allocated_shares[payer] += diff

        elif split_type == "weighted":
            assert shares is not None and sum(shares.values()) == 100, "Weights must sum to 100."
            allocated_shares = {
                user: np.round(amount * (shares[user] / 100.0), 2) 
                for user in participants
            }
            diff = round(amount - sum(allocated_shares.values()), 2)
            allocated_shares[payer] += diff

        elif split_type == "exact":
            assert shares is not None and round(sum(shares.values()), 2) == round(amount, 2), \
                "Exact shares must sum to total amount."
            allocated_shares = shares.copy()

        else:
            raise ValueError(f"Unsupported split type: {split_type}")

        # Store in transaction ledger
        new_txn = pd.DataFrame([{
            "txn_id": txn_id,
            "date": pd.to_datetime(date),
            "description": description,
            "payer": payer,
            "amount": float(amount),
            "split_type": split_type,
            "category": category
        }])
        self.transactions_df = pd.concat([self.transactions_df, new_txn], ignore_index=True)

        # Store in shares ledger
        new_splits = pd.DataFrame([
            {"txn_id": txn_id, "user": user, "share_amount": share}
            for user, share in allocated_shares.items()
        ])
        self.splits_df = pd.concat([self.splits_df, new_splits], ignore_index=True)

    def calculate_net_balances(self) -> pd.Series:
        """
        Computes net balances: Net = Total Paid - Total Consumed.
        """
        paid = self.transactions_df.groupby("payer")["amount"].sum()
        consumed = self.splits_df.groupby("user")["share_amount"].sum()
        
        all_users = list(set(paid.index).union(set(consumed.index)))
        balances = pd.Series(0.0, index=all_users)
        
        for u in all_users:
            balances[u] = round(paid.get(u, 0.0) - consumed.get(u, 0.0), 2)
            
        return balances

    def simplify_debts(self) -> List[Dict[str, Any]]:
        """
        Greedy Min-Cash-Flow resolution to simplify N-way debts to <= N-1 transactions.
        """
        balances = self.calculate_net_balances().to_dict()
        debtors = []   # (user, abs(amount))
        creditors = [] # (user, amount)

        for user, bal in balances.items():
            if bal < -0.01:
                debtors.append([user, abs(bal)])
            elif bal > 0.01:
                creditors.append([user, bal])

        settlements = []

        i, j = 0, 0
        while i < len(debtors) and j < len(creditors):
            debtor, debt_amt = debtors[i]
            creditor, cred_amt = creditors[j]

            settled_amt = min(debt_amt, cred_amt)
            settlements.append({
                "from": debtor,
                "to": creditor,
                "amount": round(settled_amt, 2)
            })

            debtors[i][1] -= settled_amt
            creditors[j][1] -= settled_amt

            if debtors[i][1] < 0.01:
                i += 1
            if creditors[j][1] < 0.01:
                j += 1

        return settlements

    def generate_visualizations(self):
        """
        Data Science Analytics & Visualizations using Matplotlib.
        """
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Category-wise Spending
        category_spend = self.transactions_df.groupby("category")["amount"].sum()
        axes[0].bar(category_spend.index, category_spend.values, color="#1a73e8", edgecolor="black")
        axes[0].set_title("Total Spend by Category", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Amount (USD)")
        axes[0].tick_params(axis='x', rotation=30)

        # 2. Consumption vs Paid per User
        paid = self.transactions_df.groupby("payer")["amount"].sum()
        consumed = self.splits_df.groupby("user")["share_amount"].sum()
        users = sorted(list(set(paid.index).union(set(consumed.index))))
        
        x = np.arange(len(users))
        width = 0.35
        
        axes[1].bar(x - width/2, [paid.get(u, 0) for u in users], width, label="Paid", color="#34a853")
        axes[1].bar(x + width/2, [consumed.get(u, 0) for u in users], width, label="Consumed", color="#ea4335")
        axes[1].set_title("Paid vs. Consumed by User", fontsize=12, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(users)
        axes[1].legend()

        # 3. Net Balances
        balances = self.calculate_net_balances()
        colors = ["#34a853" if b >= 0 else "#ea4335" for b in balances.values]
        axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
        axes[2].bar(balances.index, balances.values, color=colors, edgecolor="black")
        axes[2].set_title("Net User Balances (+ Creditor / - Debtor)", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("Net Balance (USD)")

        plt.tight_layout()
        plt.savefig("expense_analytics.png", dpi=300)
        plt.show()

# ----------------- Execution & Sample Run -----------------
if __name__ == "__main__":
    system = ExpenseSharingSystem()

    # Seed realistic Google Pay group transaction data
    system.add_expense("TX1", "2026-03-01", "Alice", 120.0, "Team Dinner", "Dining", ["Alice", "Bob", "Charlie", "Diana"], "equal")
    system.add_expense("TX2", "2026-03-02", "Bob", 60.0, "Groceries", "Household", ["Alice", "Bob"], "equal")
    system.add_expense("TX3", "2026-03-03", "Charlie", 150.0, "Airbnb Stay", "Travel", ["Alice", "Bob", "Charlie", "Diana"], "weighted", {"Alice": 20, "Bob": 20, "Charlie": 30, "Diana": 30})
    system.add_expense("TX4", "2026-03-04", "Diana", 40.0, "Movie Tickets", "Entertainment", ["Alice", "Diana"], "exact", {"Alice": 20.0, "Diana": 20.0})

    print("=== NET BALANCES ===")
    net_bals = system.calculate_net_balances()
    print(net_bals.to_string())

    print("\n=== MINIMIZED DEBT SETTLEMENT PLAN ===")
    plan = system.simplify_debts()
    for step in plan:
        print(f"• {step['from']} pays {step['to']} ${step['amount']:.2f}")

    # Generate visual analytics dashboard
    system.generate_visualizations()
