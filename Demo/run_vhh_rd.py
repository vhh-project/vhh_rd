import vhh_rd.vhh_rd as rd

config_path = "./config/config_rd.yaml"

def main():
    RD = rd.RD(config_path)
    RD.run()

if __name__ == "__main__":
    main()
