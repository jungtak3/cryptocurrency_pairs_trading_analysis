#!/usr/bin/env python3
"""
Simple script to generate markdown and PDF reports from cryptocurrency analysis
"""

from markdown_to_pdf_generator import MarkdownPDFGenerator
from cryptocurrency_pairs_trading_analysis import run_analysis

def main():
    print("=== Cryptocurrency Pairs Trading Report Generator ===\n")
    
    # Create the report generator
    generator = MarkdownPDFGenerator(output_dir="reports")
    
    try:
        # Generate the complete report
        print("Starting analysis and report generation...")
        report_path = generator.generate_report()
        
        print(f"\n✅ Success! Report generated at: {report_path}")
        print("\nTo convert to PDF manually (if pandoc failed):")
        print("1. Install pandoc: brew install pandoc basictex (macOS) or sudo apt-get install pandoc texlive-xetex (Ubuntu)")
        print(f"2. Run: pandoc {report_path} -o report.pdf --pdf-engine=xelatex -V geometry:margin=1in --toc")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()