#!/usr/bin/env python3
"""
Custom Date Range Analysis for Cryptocurrency Pairs Trading
Allows user to specify end date and automatically generates 6-month panels working backwards

Usage:
    python examples/analyze_custom_dates.py 20250430
    python examples/analyze_custom_dates.py 20241231
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cryptocurrency_pairs_trading_analysis import run_analysis
from markdown_to_pdf_generator import MarkdownPDFGenerator

def parse_date_string(date_str):
    """Parse date string in YYYYMMDD format to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYYMMDD (e.g., 20250430)")

def generate_panels(end_date):
    """
    Generate 4 panels of 6-month periods working backwards from end_date
    
    Args:
        end_date (datetime): The end date for Panel D
        
    Returns:
        dict: Panel definitions with start and end dates
    """
    
    # Calculate start of Panel D (6 months before end_date)
    panel_d_start = end_date - relativedelta(months=6) + timedelta(days=1)
    
    # Calculate Panel C (6 months before Panel D)
    panel_c_end = panel_d_start - timedelta(days=1)
    panel_c_start = panel_c_end - relativedelta(months=6) + timedelta(days=1)
    
    # Calculate Panel B (6 months before Panel C)
    panel_b_end = panel_c_start - timedelta(days=1)
    panel_b_start = panel_b_end - relativedelta(months=6) + timedelta(days=1)
    
    # Calculate Panel A (6 months before Panel B)
    panel_a_end = panel_b_start - timedelta(days=1)
    panel_a_start = panel_a_end - relativedelta(months=6) + timedelta(days=1)
    
    panels = {
        "Panel A": (panel_a_start.strftime('%Y-%m-%d'), panel_a_end.strftime('%Y-%m-%d')),
        "Panel B": (panel_b_start.strftime('%Y-%m-%d'), panel_b_end.strftime('%Y-%m-%d')),
        "Panel C": (panel_c_start.strftime('%Y-%m-%d'), panel_c_end.strftime('%Y-%m-%d')),
        "Panel D": (panel_d_start.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
    }
    
    return panels

def modify_analysis_for_custom_dates(panels):
    """
    Temporarily modify the PANELS constant in the analysis module
    """
    import cryptocurrency_pairs_trading_analysis as analysis_module
    
    # Store original panels
    original_panels = analysis_module.PANELS.copy()
    
    # Update with custom panels
    analysis_module.PANELS.clear()
    analysis_module.PANELS.update(panels)
    
    return original_panels

def restore_original_panels(original_panels):
    """Restore the original PANELS constant"""
    import cryptocurrency_pairs_trading_analysis as analysis_module
    analysis_module.PANELS.clear()
    analysis_module.PANELS.update(original_panels)

def print_panel_summary(panels):
    """Print a summary of the generated panels"""
    print("\n=== Generated Panel Dates ===")
    for panel_name, (start_date, end_date) in panels.items():
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        duration = (end_dt - start_dt).days
        print(f"{panel_name}: {start_date} to {end_date} ({duration + 1} days)")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Run cryptocurrency pairs trading analysis with custom date ranges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/analyze_custom_dates.py 20250430
    python examples/analyze_custom_dates.py 20241231
    
Notes:
    - End date will be used as Panel D end date
    - Panels A, B, C will be generated working backwards in 6-month intervals
    - Each panel covers exactly 6 months
        """
    )
    
    parser.add_argument(
        'end_date',
        help='End date for Panel D in YYYYMMDD format (e.g., 20250430)'
    )
    
    parser.add_argument(
        '--output-suffix',
        default='',
        help='Optional suffix for output files (e.g., "_custom")'
    )
    
    args = parser.parse_args()
    
    try:
        # Parse and validate end date
        end_date = parse_date_string(args.end_date)
        print(f"Analysis end date: {end_date.strftime('%Y-%m-%d')}")
        
        # Generate panel dates
        panels = generate_panels(end_date)
        print_panel_summary(panels)
        
        # Validate that we have reasonable date ranges
        earliest_date = datetime.strptime(panels["Panel A"][0], '%Y-%m-%d')
        latest_date = datetime.strptime(panels["Panel D"][1], '%Y-%m-%d')
        total_duration = (latest_date - earliest_date).days
        
        if total_duration < 365:
            print("⚠️  Warning: Total analysis period is less than 1 year")
        if earliest_date.year < 2017:
            print("⚠️  Warning: Analysis starts before 2017 (cryptocurrency data may be limited)")
            
        # Modify analysis module with custom dates
        print("Updating analysis configuration with custom dates...")
        original_panels = modify_analysis_for_custom_dates(panels)
        
        try:
            # Run the analysis
            print("Starting cryptocurrency pairs trading analysis...")
            results = run_analysis()
            
            # Generate report with custom suffix
            output_suffix = args.output_suffix or f"_custom_{args.end_date}"
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
            
            generator = MarkdownPDFGenerator(output_dir=output_dir, custom_panels=panels)
            
            # Customize the output filename
            original_timestamp = generator.timestamp
            generator.timestamp = original_timestamp + output_suffix
            
            print("Generating markdown report...")
            report_path = generator.generate_report(lambda: results)
            
            print(f"\n✅ Success! Custom date analysis complete!")
            print(f"📊 Report generated: {report_path}")
            print(f"📅 Analysis period: {panels['Panel A'][0]} to {panels['Panel D'][1]}")
            
            # Print summary statistics
            if results.get('table1'):
                panel_count = len(results['table1'])
                print(f"📈 Panels analyzed: {panel_count}")
                
            if results.get('table2'):
                first_panel = next(iter(results['table2'].values()))
                if not first_panel.empty:
                    pair_count = len(first_panel.columns)
                    print(f"💱 Pairs analyzed: {pair_count}")
            
            print("\nTo convert to PDF:")
            print(f"pandoc {report_path} -o custom_analysis.pdf --pdf-engine=xelatex -V geometry:margin=1in --toc")
            
        finally:
            # Always restore original panels
            restore_original_panels(original_panels)
            
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()