import unittest
from datetime import datetime
import pytz
from market_indices_analyzer import get_market_hours, get_us_market_hours

class TestMarketHours(unittest.TestCase):
    def test_india_market_hours_weekday_before_open(self):
        """Test Indian market hours on a weekday before market opens"""
        test_time = datetime(2024, 12, 9, 8, 0)  # 8:00 AM IST
        hours = get_market_hours(test_time)
        self.assertEqual(hours['status'], 'CLOSED')
        self.assertEqual(hours['trading_hours'], '09:15 - 15:30 IST')
        self.assertEqual(hours['next_open'], '2024-12-09 09:15 IST')
    
    def test_india_market_hours_weekday_during_market(self):
        """Test Indian market hours during market hours"""
        test_time = datetime(2024, 12, 9, 13, 0)  # 1:00 PM IST
        hours = get_market_hours(test_time)
        self.assertEqual(hours['status'], 'OPEN')
        self.assertEqual(hours['trading_hours'], '09:15 - 15:30 IST')
        self.assertEqual(hours['next_open'], '2024-12-09 09:15 IST')
    
    def test_india_market_hours_weekday_after_close(self):
        """Test Indian market hours after market closes"""
        test_time = datetime(2024, 12, 9, 16, 0)  # 4:00 PM IST
        hours = get_market_hours(test_time)
        self.assertEqual(hours['status'], 'CLOSED')
        self.assertEqual(hours['trading_hours'], '09:15 - 15:30 IST')
        self.assertEqual(hours['next_open'], '2024-12-10 09:15 IST')
    
    def test_india_market_hours_weekend(self):
        """Test Indian market hours on weekend"""
        test_time = datetime(2024, 12, 7, 12, 0)  # Saturday 12:00 PM IST
        hours = get_market_hours(test_time)
        self.assertEqual(hours['status'], 'CLOSED')
        self.assertEqual(hours['trading_hours'], '09:15 - 15:30 IST')
        self.assertEqual(hours['next_open'], '2024-12-09 09:15 IST')
    
    def test_us_market_hours_weekday_before_open(self):
        """Test US market hours on a weekday before market opens"""
        test_time = datetime(2024, 12, 9, 8, 0)  # 8:00 AM EST
        hours = get_us_market_hours(test_time)
        self.assertEqual(hours['status'], 'CLOSED')
        self.assertEqual(hours['trading_hours'], '09:30 - 16:00 EST')
        self.assertEqual(hours['next_open'], '2024-12-09 09:30 EST')
    
    def test_us_market_hours_weekday_during_market(self):
        """Test US market hours during market hours"""
        test_time = datetime(2024, 12, 9, 13, 0)  # 1:00 PM EST
        hours = get_us_market_hours(test_time)
        self.assertEqual(hours['status'], 'OPEN')
        self.assertEqual(hours['trading_hours'], '09:30 - 16:00 EST')
        self.assertEqual(hours['next_open'], '2024-12-09 09:30 EST')
    
    def test_us_market_hours_weekday_after_close(self):
        """Test US market hours after market closes"""
        test_time = datetime(2024, 12, 9, 17, 0)  # 5:00 PM EST
        hours = get_us_market_hours(test_time)
        self.assertEqual(hours['status'], 'CLOSED')
        self.assertEqual(hours['trading_hours'], '09:30 - 16:00 EST')
        self.assertEqual(hours['next_open'], '2024-12-10 09:30 EST')
    
    def test_us_market_hours_weekend(self):
        """Test US market hours on weekend"""
        test_time = datetime(2024, 12, 7, 12, 0)  # Saturday 12:00 PM EST
        hours = get_us_market_hours(test_time)
        self.assertEqual(hours['status'], 'CLOSED')
        self.assertEqual(hours['trading_hours'], '09:30 - 16:00 EST')
        self.assertEqual(hours['next_open'], '2024-12-09 09:30 EST')

if __name__ == '__main__':
    unittest.main()
