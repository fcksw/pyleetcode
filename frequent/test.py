def custom_crc32(data):
    # 初始化寄存器为全 1
    crc = 0xFFFFFFFF
    # 生成多项式
    polynomial = 0x04C11DB7
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
    # 取反操作
    return crc ^ 0xFFFFFFFF

# 示例数据
data = b"Hello, World!"
# 计算 CRC32 校验值
crc_value = custom_crc32(data)
print(f"自定义 CRC32 校验值: {hex(crc_value)}")
print(len("yYM9EYMYyy9hOpyH"))


if __name__ == "__main__":
    def custom_crc32(data):
        # 初始化寄存器为全 1
        crc = 0xFFFFFFFF
        # 生成多项式
        polynomial = 0x04C11DB7
        for byte in data:
            crc ^= byte << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ polynomial
                else:
                    crc <<= 1
        # 取反操作
        return crc ^ 0xFFFFFFFF
    # 示例数据
    data = b"Hello, World!"
    # 计算 CRC32 校验值
    crc_value = custom_crc32(data)
    print(f"自定义 CRC32 校验值: {hex(crc_value)}")

    print(len('0xb5391ccfffcaf71587c688fe60e6d8fedf'))