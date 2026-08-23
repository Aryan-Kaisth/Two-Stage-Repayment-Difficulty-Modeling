## About Home Credit

Home Credit is a consumer finance company focused on expanding access to credit for people who may have limited or no traditional credit history. To assess creditworthiness and support responsible lending, it uses a broader range of financial and behavioral information beyond conventional credit data. The Home Credit dataset reflects this approach by combining current loan applications with previous credit history, account-level records, and detailed repayment behavior.

<p align="left">
  <img src="../app/assets/Home-Credit-Logo.png" alt="Home Credit Logo" width="60%">
</p>

## About the Dataset

The Home Credit dataset is a collection of interconnected tables representing different parts of an applicant's credit journey. It includes information about current loan applications, previous applications, credits reported by other financial institutions, monthly account balances, credit card activity, and installment payments.

The tables are linked through common identifiers and operate at different levels of granularity, allowing a single applicant to be connected to multiple previous credits, monthly account records, and payment histories.

## Dataset Files

The dataset is divided into eight related files, each capturing a different level of an applicant's credit history.

| File | Granularity | What it contains |
|---|---|---|
| `application_train.csv` | One row per current application | Current loan applications along with the `TARGET` repayment outcome. |
| `application_test.csv` | One row per current application | Current loan applications used for prediction; does not contain `TARGET`. |
| `bureau.csv` | One row per previous external credit | Credits previously held with other financial institutions and reported to the Credit Bureau. |
| `bureau_balance.csv` | One row per credit per month | Monthly status and balance history for credits recorded in `bureau.csv`. |
| `previous_application.csv` | One row per previous application | Historical loan applications previously made by the applicant with Home Credit. |
| `POS_CASH_balance.csv` | One row per previous loan per month | Monthly snapshots of previous POS and cash loans held with Home Credit. |
| `credit_card_balance.csv` | One row per previous credit card per month | Monthly snapshots of previous Home Credit credit-card accounts. |
| `installments_payments.csv` | One row per payment/installment | Historical repayment records for previously disbursed Home Credit loans, including missed payments. |

At a high level, the data follows this structure:

```text
Current Application
       │
       ├── External Credit History
       │       └── Monthly Bureau Balance
       │
       └── Previous Home Credit Applications
               ├── POS / Cash Loan History
               ├── Credit Card History
               └── Installment Payments
```

> **Detailed feature descriptions:**  
> For the complete list of columns and their definitions, see [`feature_description.txt`](../docs/feature_description.txt).

## Data Structure

The relationships between the datasets are illustrated in the ER diagram below.

<p align="left">
  <img src="../docs/ER_diagram.png" alt="Home Credit Dataset ER Diagram" width="70%">
</p>

## Primary & Foreign Keys

The tables in the Home Credit dataset are connected through a set of identifiers that allow information from different levels of the customer's credit history to be linked together.

A **primary key (PK)** uniquely identifies a record within a table, while a **foreign key (FK)** is an identifier used in another table to reference that record. In this dataset, some identifiers are unique within a table, while others intentionally appear multiple times because the relationship is one-to-many.

### Key Identifiers

| Identifier     | Role                                                                                                  | Used in                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `SK_ID_CURR`   | **Primary key** for a current application; also used as a foreign key in related historical tables    | `application_train`, `application_test`, `bureau`, `previous_application`                  |
| `SK_ID_BUREAU` | **Primary key** for a Bureau credit record                                                            | `bureau` → `bureau_balance`                                                                |
| `SK_ID_PREV`   | **Primary key** for a previous Home Credit application/credit; used to connect its historical records | `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments` |

The most important relationship starts with **`SK_ID_CURR`**, which identifies the current applicant/loan and links it to their previous credit history and previous Home Credit applications. Those historical records can then be connected to their own monthly account or payment-level data using **`SK_ID_BUREAU`** or **`SK_ID_PREV`**.

This creates a hierarchical structure where a single current application can be associated with **multiple previous credits, multiple monthly balance records, and multiple repayment records**.

## Dataset Source

The dataset is from the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk) on Kaggle.