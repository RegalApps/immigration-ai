#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pytest
import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, Any
import subprocess
import psutil
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "server_url": "http://localhost:8002",  # Match the server's port
    "startup_timeout": 10,  # seconds
    "response_timeout": 30,  # seconds
    "max_memory_usage": 1024 * 1024 * 1024,  # 1GB
    "max_response_time": 5,  # seconds
    "concurrent_users": 5,
    "requests_per_user": 10,
    "expected_success_rate": 0.95
}

class ChatbotTestSuite:
    """End-to-end test suite for the Immigration Chatbot."""
    
    def __init__(self):
        self.server_process = None
        self.start_time = None
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "memory_usage": [],
            "errors": []
        }

    async def start_server(self):
        """Start the chatbot server as a separate process."""
        try:
            # Start server using subprocess
            self.server_process = subprocess.Popen(
                ["python3", "immigration_chatbot.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for server to output its port
            port = None
            start_wait = time.time()
            while time.time() - start_wait < TEST_CONFIG["startup_timeout"]:
                line = self.server_process.stdout.readline()
                if "Starting server on port" in line:
                    port = int(line.strip().split()[-1])
                    break
                
                # Check for early failure
                if self.server_process.poll() is not None:
                    err = self.server_process.stderr.read()
                    raise Exception(f"Server failed to start: {err}")
            
            if not port:
                raise Exception("Could not determine server port")
                
            # Update server URL with actual port
            TEST_CONFIG["server_url"] = f"http://localhost:{port}"
            logger.info(f"Server started on port {port}")
            
            # Wait for server to be ready
            async with aiohttp.ClientSession() as session:
                while time.time() - start_wait < TEST_CONFIG["startup_timeout"]:
                    try:
                        async with session.get(TEST_CONFIG["server_url"]) as response:
                            if response.status == 200:
                                logger.info("Server ready to accept connections")
                                return True
                    except:
                        await asyncio.sleep(0.5)
            
            raise Exception("Server failed to become ready within timeout")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the chatbot server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
            logger.info("Server stopped")

    async def send_chat_message(self, message: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Send a chat message and return the response."""
        try:
            start_time = time.time()
            async with session.post(
                f"{TEST_CONFIG['server_url']}/chat",
                json={"message": message}
            ) as response:
                response_time = time.time() - start_time
                self.metrics["response_times"].append(response_time)
                
                if response.status == 200:
                    self.metrics["successful_requests"] += 1
                    return {
                        "success": True,
                        "response_time": response_time,
                        "status": response.status
                    }
                else:
                    self.metrics["failed_requests"] += 1
                    error_msg = await response.text()
                    self.metrics["errors"].append(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "status": response.status
                    }
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.metrics["errors"].append(str(e))
            return {"success": False, "error": str(e)}

    def monitor_resources(self):
        """Monitor system resources."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics["memory_usage"].append(memory_info.rss)
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            self.metrics["memory_usage"].append(0)  # Fallback value

    async def run_load_test(self):
        """Run concurrent load test."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(TEST_CONFIG["concurrent_users"]):
                for i in range(TEST_CONFIG["requests_per_user"]):
                    message = f"Test message {i} from concurrent load test"
                    task = asyncio.create_task(self.send_chat_message(message, session))
                    tasks.append(task)
                    self.metrics["total_requests"] += 1
                    # Monitor resources after each request
                    self.monitor_resources()
            
            await asyncio.gather(*tasks)

    def generate_report(self):
        """Generate test results report."""
        success_rate = self.metrics["successful_requests"] / self.metrics["total_requests"]
        avg_response_time = np.mean(self.metrics["response_times"])
        max_response_time = np.max(self.metrics["response_times"])
        p95_response_time = np.percentile(self.metrics["response_times"], 95)
        max_memory = max(self.metrics["memory_usage"]) / (1024 * 1024)  # MB

        report = f"""
=== Immigration Chatbot QA Test Report ===
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
-------------------
Total Requests: {self.metrics['total_requests']}
Success Rate: {success_rate:.2%}
Average Response Time: {avg_response_time:.2f}s
95th Percentile Response Time: {p95_response_time:.2f}s
Maximum Response Time: {max_response_time:.2f}s
Peak Memory Usage: {max_memory:.2f}MB

Error Summary:
-------------
Total Errors: {len(self.metrics['errors'])}
{"No errors encountered" if not self.metrics['errors'] else 'Last 5 errors:'}
{chr(10).join(self.metrics['errors'][-5:]) if self.metrics['errors'] else ''}

Test Result: {'PASS' if self._check_test_criteria() else 'FAIL'}
"""
        logger.info(report)
        return report

    def _check_test_criteria(self) -> bool:
        """Check if all test criteria are met."""
        criteria = [
            (self.metrics["successful_requests"] / self.metrics["total_requests"]) >= TEST_CONFIG["expected_success_rate"],
            np.mean(self.metrics["response_times"]) <= TEST_CONFIG["max_response_time"],
            max(self.metrics["memory_usage"]) <= TEST_CONFIG["max_memory_usage"]
        ]
        return all(criteria)

    async def run_functional_tests(self):
        """Run functional test cases."""
        test_cases = [
            {
                "message": "What visa options are available for software engineers?",
                "expected_keywords": ["H-1B", "O-1", "software", "engineer"]
            },
            {
                "message": "How long does H1B processing take?",
                "expected_keywords": ["process", "time", "H-1B", "USCIS"]
            },
            {
                "message": "Generate a timeline for my H1B application",
                "expected_keywords": ["timeline", "application", "process", "steps"]
            }
        ]

        async with aiohttp.ClientSession() as session:
            for case in test_cases:
                result = await self.send_chat_message(case["message"], session)
                if not result["success"]:
                    logger.error(f"Functional test failed for: {case['message']}")
                    self.metrics["errors"].append(f"Functional test failed: {case['message']}")

async def main():
    """Main test execution function."""
    test_suite = ChatbotTestSuite()
    
    try:
        # Start server
        logger.info("Starting server...")
        if not await test_suite.start_server():
            logger.error("Failed to start server. Aborting tests.")
            return

        # Run functional tests
        logger.info("Running functional tests...")
        await test_suite.run_functional_tests()

        # Run load tests
        logger.info("Running load tests...")
        await test_suite.run_load_test()

        # Generate report
        report = test_suite.generate_report()
        
        # Save report to file
        with open("qa_test_report.txt", "w") as f:
            f.write(report)

    finally:
        # Cleanup
        test_suite.stop_server()

if __name__ == "__main__":
    asyncio.run(main())
