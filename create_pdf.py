from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt="""Title: Insurance Policy Overview

This document outlines the key points of insurance policy ABC123.

1. Coverage:
   - Health: Included
   - Dental: Included
   - Vision: Optional

2. Policy Start Date: January 1, 2024
3. Policy End Date: December 31, 2024
4. Annual Premium: $1,200

Terms:
This policy is subject to terms and conditions outlined in the agreement. Please refer to section 4 for claim procedures.
""")

pdf.output("insurance_policy.pdf")
