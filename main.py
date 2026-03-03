import ast
import contextlib
import io
import ollama
import pandas as pd
import re

# ── 1. Load Excel ─────────────────────────────────────────────────────────────
df = pd.read_excel("SalesDataAnalysisV3.xlsx")
df["TotalSales"] = df["Quantity"] * df["SalePrice"]
df["OrderHour"] = pd.to_datetime(
    df["Order Time Modified"], format="%I:%M %p", errors="coerce"
).dt.hour
df["WeekNumber"] = pd.to_numeric(
    df["Week Slot"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
)

# ── 2. Build System Prompt with actual column info ────────────────────────────
def build_system_prompt(df):
    columns     = list(df.columns)
    dtypes      = df.dtypes.to_string()
    sample_data = df.head(2).to_string()
    time_cols   = [c for c in df.columns if re.search(r"time|hour|timestamp|date", c, re.IGNORECASE)]

    return f"""
You are a Python Pandas data analyst.
You will be given a question about a sales Excel dataset loaded as a pandas DataFrame called 'df'.

DATAFRAME INFO:
Columns: {columns}
Likely Time Columns: {time_cols}

Data Types:
{dtypes}

Sample Data:
{sample_data}

YOUR JOB:
- Write a small Python code snippet to answer the user's question using 'df'
- Store the final answer in a variable called 'result'
- Always print the result at the end using print()
- Keep code simple and clean

RULES:
* Return ONLY Python code
* NO explanations
* NO markdown formatting
* NO code blocks or triple backticks
* Use only pandas operations on 'df'
* Do NOT write to disk or access network
* If question is unclear, print a helpful message
* For currency values, format as ₹ with 2 decimal places
* For time/week filters, prefer helper columns: 'OrderHour' and 'WeekNumber'
* If datetime parsing is needed, always use explicit format=... and errors='coerce'
* Always use exact column names from the Columns list above (do not invent columns)

EXAMPLES:

Question: What is the total sales?
Code:
result = df["TotalSales"].sum()
print(f"Total Sales: ₹{{result:,.2f}}")

Question: Which day has the highest sales?
Code:
result = df.groupby("OrderDay")["TotalSales"].sum().idxmax()
print(f"Highest Sales Day: {{result}}")

Question: How many orders were placed in each category?
Code:
result = df.groupby("CategoryName")["OrderNumber"].nunique()
print(result.to_string())

Question: What is the average discount given?
Code:
result = df["DiscountInPercentage"].mean()
print(f"Average Discount: {{result:.2f}}%")

Question: Which item sold most between 10am and 12pm?
Code:
filtered = df[(df["OrderHour"] >= 10) & (df["OrderHour"] < 12)]
result = filtered.groupby("ItemName")["Quantity"].sum().idxmax()
print(f"Top Item (10-12): {{result}}")

Question: Which item sold most between 10am and 12pm in week14?
Code:
filtered = df[
    (df["OrderHour"] >= 10)
    & (df["OrderHour"] < 12)
    & (df["WeekNumber"] == 14)
]
result = filtered.groupby("ItemName")["Quantity"].sum().idxmax()
print(f"Top Item (10-12, Week-14): {{result}}")

"""

# ── 3. Ask Ollama to generate Pandas code ─────────────────────────────────────
def ask_ollama(user_question, df, model="llama3:8b", extra_context=None):
    system_prompt = build_system_prompt(df)
    user_content = user_question
    if extra_context:
        user_content = f"{user_question}\n\n{extra_context}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ],
        options={"temperature": 0.0}
    )

    raw_code = response["message"]["content"].strip()

    # Clean up if model wraps in markdown code blocks anyway
    raw_code = re.sub(r"```python", "", raw_code, flags=re.IGNORECASE)
    raw_code = re.sub(r"```",       "", raw_code)

    # Remove non-code chatter (e.g., "Here's the code:")
    lines = [ln.rstrip() for ln in raw_code.splitlines()]
    code_lines = []
    started = False
    for ln in lines:
        if not started:
            if re.match(r"^\s*(#|result\s*=|print\s*\(|df[\.\[]|pd\.)", ln):
                started = True
        if started and ln.strip() != "":
            code_lines.append(ln)
    filtered_lines = []
    for ln in code_lines:
        if re.match(r"^\s*(#|result\s*=|print\s*\(|df[\.\[]|pd\.|[A-Za-z_][A-Za-z0-9_]*\s*=)", ln):
            filtered_lines.append(ln)
    cleaned = "\n".join(filtered_lines).strip()

    return cleaned if cleaned else raw_code.strip()

# ── 4. Safely execute the generated code ─────────────────────────────────────
class SafeCodeVisitor(ast.NodeVisitor):
    BLOCKED_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.With,
        ast.Try,
        ast.Raise,
        ast.Global,
        ast.Nonlocal,
        ast.Lambda,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )

    BLOCKED_CALLS = {"eval", "exec", "open", "compile", "input", "__import__"}

    def visit(self, node):
        if isinstance(node, self.BLOCKED_NODES):
            raise ValueError(f"Blocked syntax: {type(node).__name__}")
        return super().visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            raise ValueError("Blocked attribute access to dunder")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("__"):
            raise ValueError("Blocked name access to dunder")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.BLOCKED_CALLS:
            raise ValueError(f"Blocked call: {node.func.id}")
        self.generic_visit(node)


def _validate_code(code):
    tree = ast.parse(code)
    SafeCodeVisitor().visit(tree)

def _validate_columns(code, df):
    # Basic static check for df["col"] and df['col'] usage
    used = re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]", code)
    missing = [c for c in used if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns used: {missing}")


def execute_code(code, df, show_code=False):
    if show_code:
        print(f"\n📝 Generated Code:\n{'-'*40}\n{code}\n{'-'*40}\n")
    print("📊 Result:")

    try:
        _validate_code(code)
        _validate_columns(code, df)
        safe_builtins = {
            "print": print,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "str": str,
            "int": int,
            "float": float,
            "abs": abs,
            "round": round,
        }
        globals_dict = {"df": df, "pd": pd, "__builtins__": safe_builtins}
        locals_dict = {}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, globals_dict, locals_dict)

        output = buf.getvalue().strip()
        if output:
            print(output)

        if "result" in locals_dict and not output:
            print(locals_dict["result"])

        if "result" not in locals_dict:
            raise ValueError("Code did not set 'result'")

        return True, None
    except Exception as e:
        print(f"⚠️ Execution Error: {e}")
        print("💡 Try rephrasing your question.")
        return False, str(e)

# ── 5. Main Loop ──────────────────────────────────────────────────────────────
def main():
    MODEL = "llama3:8b"  

    print(f"🤖 AI Excel Analyst Ready! (Ollama → {MODEL})")
    print(f"📂 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")
    print("\nYou can ask ANYTHING about your data!")
    print("Type 'exit' to quit | Type 'columns' to see all columns\n")

    while True:
        user_input = input("❓ Your Question: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        if user_input.lower() == "columns":
            print(f"\n📋 Available Columns:\n{list(df.columns)}\n")
            continue

        try:
            print("⏳ Thinking...")
            code = ask_ollama(user_input, df, model=MODEL)
            ok, err = execute_code(code, df, show_code=False)

            if not ok:
                print("🔁 Retrying with error feedback...")
                extra = (
                    "The previous code failed. Fix it and return only corrected Python code.\n"
                    f"Error: {err}\n"
                    "Use exact column names from the Columns list above. "
                    "For time filtering, use 'OrderHour'. For week filtering, use 'WeekNumber'."
                )
                retry_code = ask_ollama(user_input, df, model=MODEL, extra_context=extra)
                execute_code(retry_code, df, show_code=False)

        except Exception as e:
            print(f"❌ Error: {e}")

        print("=" * 50)

if __name__ == "__main__":
    main()
