# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import pathlib
Python_Lib_path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(Python_Lib_path)
from My_Lib_Stock import *


class SSH_Account:
    def __init__(self, input_str: str):
        r"""
        :param input_str: name 100.100.101.123:1823 username password
                        or with ssh private key name 100.100.101.123:1234 username E:\path_to_key_file\keyfile.openssh
        """
        input_str = input_str.strip().split(" ")
        self.tag = input_str[0]
        self.ip_port = input_str[1]
        self.ip, self.port = self.ip_port.split(":")
        
        if is_int(self.port):
            self.port = int(self.port)
        else:
            self.port = 22

        self.username = input_str[2]
        self.password = input_str[3]

    def __str__(self):
        return self.username + ' @ ' + self.tag


def download_sftp_file(ssh_account: SSH_Account, remote_filepath, local_filepath, transport_object=None, sftp_object=None):
    # 产生一个随机的临时文件，然后改名为想要的文件名，某种程度上保证原子性
    remove_append = filename_class(local_filepath).only_remove_append
    append = filename_class(local_filepath).append

    local_temp_filepath = remove_append + "_TEMP_For_atomicity." + append

    import paramiko
    if not transport_object:
        transport = paramiko.Transport((ssh_account.ip, ssh_account.port))
        transport.connect(username=ssh_account.username, password=ssh_account.password)
    else:
        transport = transport_object

    if not sftp_object:
        sftp = paramiko.SFTPClient.from_transport(transport)
    else:
        sftp = sftp_object

    ssh_for_homedir = paramiko.SSHClient()
    ssh_for_homedir.load_system_host_keys()
    ssh_for_homedir.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_for_homedir.connect(ssh_account.ip, ssh_account.port, username=ssh_account.username, password=ssh_account.password)

    stdin, stdout, stderr = ssh_for_homedir.exec_command("echo " + remote_filepath)
    remote_filepath = stdout.read().decode('utf-8').strip()

    sftp.get(remote_filepath, local_temp_filepath)

    if os.path.isfile(local_filepath):
        os.remove(local_filepath)
    os.rename(local_temp_filepath, local_filepath)

    if not sftp_object:
        sftp.close()

    ssh_for_homedir.close()


def open_url_with_chrome(url):
    import webbrowser
    webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"))
    webbrowser.get('chrome').open_new_tab(url)


def open_phantomJS(url, use_chrome=False):
    from selenium import webdriver
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    
    if use_chrome:
        test_driver = webdriver.Chrome()
    else:
        # Connect_to_PhantomJS
        dcap = dict(DesiredCapabilities.PHANTOMJS)
        dcap["phantomjs.page.settings.userAgent"] = \
            "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        test_driver = webdriver.PhantomJS(desired_capabilities=dcap)

    test_driver.get(url)
    return test_driver


def urlopen_inf_retry(url, prettify=True, use_cookie=False, retry_limit=100, opener=None, timeout=60, print_out=True):
    from bs4 import BeautifulSoup

    from http.cookiejar import CookieJar
    import urllib
    from urllib import request
    import socket

    html = None

    def request_page(request_url, request_page_opener, request_page_timeout=60):
        if use_cookie:
            if not request_page_opener:
                request_page_opener = request.build_opener(request.HTTPCookieProcessor(CookieJar()))
            return request_page_opener.open(request_url, timeout=request_page_timeout)
        else:
            req = request.Request(request_url, headers={
                'User-Agent': "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"})
            return request.urlopen(req, timeout=request_page_timeout)

    fail = True
    retry_count = 1
    while fail and retry_count <= retry_limit:
        if print_out:
            print("Requesting:", url, end='\t')
        try:
            html = request_page(url, opener, request_page_timeout=timeout).read()
            fail = False
            if print_out:
                print("Request Finished.")
        except (socket.gaierror, urllib.error.URLError, ConnectionResetError, TimeoutError, socket.timeout, UnboundLocalError) as e:
            if print_out:
                print('\nURL open failure. Retrying... ' + str(retry_count) + '/' + str(retry_limit), e)
            retry_count += 1
            import time
            time.sleep(2)

    if not html:
        return ""
    if prettify:
        html = BeautifulSoup(html, "lxml").prettify()
        return BeautifulSoup(html, "lxml")
    else:
        return html


def match_attr_bs4(bs_object, key, value, match_full=False):
    # time.sleep(1)
    if not match_full:
        if bs_object.get(key):
            if value in bs_object[key]:
                return True
    else:
        if bs_object.get(key):
            target = bs_object[key]
            target = " ".join([" ".join(x.split()) for x in target])
            value = " ".join(value.split)
            if target == value:
                return True
    return False


def open_tab(url):
    import webbrowser
    webbrowser.register('brave', None, webbrowser.BackgroundBrowser(r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"))
    browser = webbrowser.get('brave')
    browser.open_new_tab(url)


if __name__ == '__main__':
    pass
