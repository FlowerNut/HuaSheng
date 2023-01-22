from collector import share_list


if __name__ == '__main__':
    # ----------download list------------------
    sl = share_list.ShareListDownloader()
    sl.download_share_list()
    print(sl.read_share_list())
