#!/usr/bin/env python3
# Jordan huang<good5dog5@gmail.com>

import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from bs4 import BeautifulSoup

if __name__ == '__main__':

    url = 'https://challenger.ai'
    submit_url = 'https://challenger.ai/competition/trendsense/submit'
    ACC = 'jordan.huang@gapp.nthu.edu.tw'
    PWD   = 'jm_1234'

    driver = webdriver.Firefox()
    driver.get(url)

    element = driver.find_element_by_id('login_btn')
    element.click()

    ACC_field = driver.find_element_by_id("ipt_account")
    PWD_field = driver.find_element_by_id("ipt_pwd")
    ACC_field.send_keys(ACC)
    PWD_field.send_keys(PWD)

    login_btn = driver.find_element_by_id('login_submit')
    login_btn.click()

    for i in range(99,116):

        # Locate submit page
        driver.get(submit_url)
        # Find file field
        file_field = driver.find_element_by_id('pred_file')

        f = "/home/jordan/Downloads/K/K" + str(i) + ".csv"
        file_field.send_keys(f)
        try:
            # 一直等到出現"得分"兩字，最多等20秒
            element = WebDriverWait(driver, 20).until(EC.text_to_be_present_in_element((By.ID, "up_result"), "得分"))
        finally:
            result = driver.find_element_by_id('up_result')
            print(i, result.text)

    driver.quit()

    
