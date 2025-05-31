#!/usr/bin/env python3
"""
Date Calculator Utility for Cryptocurrency Pairs Trading Analysis
Helps calculate and visualize date ranges for custom analysis periods
"""

import sys
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def parse_date_string(date_str):
    """Parse date string in YYYYMMDD format to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYYMMDD (e.g., 20250430)")

def generate_panels(end_date):
    """Generate 4 panels of 6-month periods working backwards from end_date"""
    
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
    
    return {
        "Panel A": (panel_a_start, panel_a_end),
        "Panel B": (panel_b_start, panel_b_end),
        "Panel C": (panel_c_start, panel_c_end),
        "Panel D": (panel_d_start, end_date),
    }

def print_calendar_view(panels):
    """Print a calendar-style view of the panels"""
    print("\n=== Calendar View ===")
    
    earliest = min(panels["Panel A"][0], panels["Panel B"][0], panels["Panel C"][0], panels["Panel D"][0])
    latest = max(panels["Panel A"][1], panels["Panel B"][1], panels["Panel C"][1], panels["Panel D"][1])
    
    print(f"Total Analysis Period: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
    print(f"Total Duration: {(latest - earliest).days + 1} days ({(latest - earliest).days / 30.44:.1f} months)")
    print()
    
    for panel_name, (start_date, end_date) in panels.items():
        duration = (end_date - start_date).days + 1
        print(f"{panel_name}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({duration} days)")
    print()

def validate_date_coverage(panels):
    """Validate that panels don't overlap and cover reasonable periods"""
    issues = []
    
    # Check for gaps or overlaps
    for i, (current_panel, next_panel) in enumerate(zip(
        ["Panel A", "Panel B", "Panel C"],
        ["Panel B", "Panel C", "Panel D"]
    )):
        current_end = panels[current_panel][1]
        next_start = panels[next_panel][0]
        gap = (next_start - current_end).days
        
        if gap != 1:
            if gap > 1:
                issues.append(f"Gap of {gap-1} days between {current_panel} and {next_panel}")
            else:
                issues.append(f"Overlap of {1-gap} days between {current_panel} and {next_panel}")
    
    # Check panel durations
    for panel_name, (start_date, end_date) in panels.items():
        duration_days = (end_date - start_date).days + 1
        duration_months = duration_days / 30.44
        
        if duration_months < 5.5 or duration_months > 6.5:
            issues.append(f"{panel_name} duration is {duration_months:.1f} months (expected ~6)")
    
    # Check if analysis period is reasonable for crypto data
    earliest = panels["Panel A"][0]
    if earliest.year < 2017:
        issues.append(f"Analysis starts in {earliest.year} - limited crypto data available before 2017")
    
    return issues

def main():
    parser = argparse.ArgumentParser(
        description='Calculate date ranges for cryptocurrency pairs trading analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/date_calculator.py 20250430
    python examples/date_calculator.py 20241231 --validate
    python examples/date_calculator.py 20230630 --format csv
        """
    )
    
    parser.add_argument(
        'end_date',
        help='End date for Panel D in YYYYMMDD format (e.g., 20250430)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the generated date ranges for potential issues'
    )
    
    parser.add_argument(
        '--format',
        choices=['table', 'csv', 'json'],
        default='table',
        help='Output format for date ranges'
    )
    
    args = parser.parse_args()
    
    try:
        # Parse end date
        end_date = parse_date_string(args.end_date)
        
        # Generate panels
        panels = generate_panels(end_date)
        
        # Display results
        if args.format == 'table':
            print(f"\n=== Date Calculation for End Date: {end_date.strftime('%Y-%m-%d')} ===")
            print_calendar_view(panels)
            
        elif args.format == 'csv':
            print("Panel,Start_Date,End_Date,Duration_Days")
            for panel_name, (start_date, end_date) in panels.items():
                duration = (end_date - start_date).days + 1
                print(f"{panel_name},{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')},{duration}")
                
        elif args.format == 'json':
            import json
            result = {}
            for panel_name, (start_date, end_date) in panels.items():
                result[panel_name] = {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_days': (end_date - start_date).days + 1
                }
            print(json.dumps(result, indent=2))
        
        # Validation
        if args.validate:
            print("=== Validation Results ===")
            issues = validate_date_coverage(panels)
            if issues:
                print("⚠️  Issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ All date ranges look good!")
            print()
        
        # Usage instructions
        if args.format == 'table':
            print("=== Usage Instructions ===")
            print(f"To run analysis with these dates:")
            print(f"   python examples/analyze_custom_dates.py {args.end_date}")
            print()
            print("Original paper periods (for comparison):")
            print("   Panel A: 2018-01-01 to 2018-06-30")
            print("   Panel B: 2018-07-01 to 2018-12-31") 
            print("   Panel C: 2019-01-01 to 2019-06-30")
            print("   Panel D: 2019-07-01 to 2019-12-31")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()