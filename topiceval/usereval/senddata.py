from __future__ import print_function
from __future__ import division

import win32com.client as win32
import shutil
import os
import logging

logger = logging.getLogger(__name__)


def makezip(dirname):
    shutil.make_archive("./data/userdata", format="zip", root_dir=dirname)
    return


def sendmail():
    try:
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        mail.To = 't-avsriv@microsoft.com'
        mail.Subject = 'Topic Model Evaluation Data'
        mail.Body = ''
        # mail.HTMLBody = ''# this field is optional
        attachment1 = os.getcwd() + "/data/userdata.zip"
        mail.Attachments.Add(Source=attachment1)
        mail.Send()
        logger.info("Mail sent")
    except Exception as e:
        logger.error("Exception when sending mail: %s" % e)
    return
