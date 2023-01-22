from collector import share_history as collector_sh


if __name__ == '__main__':
    # ----------download history---------------
    csh = collector_sh.ShareHistoryDownloader()
    csh.download_share_history()

