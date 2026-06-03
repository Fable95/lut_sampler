import argparse

def f(k):
    return 2**k - k - 1

def cost_all(k_l, w):
    seen_zero = False
    sum_k = sum(k_l)
    sum_max = 2**(sum_k)
    for k in k_l:
        if k == 0:
            seen_zero = True
        elif seen_zero and k != 0:
            return (sum_max, sum_max, sum_max)
    prep = sum([f(k) for k in k_l])
    # first dimension is free
    total = prep
    for (i,k) in enumerate(k_l):
        if i == 0:
            continue
        online = 1
        for j in range(i+1, len(k_l)):
            online *= 2**k_l[j]
        if k > 0:
            total += w * online
        # print(k)
    total += sum_k
    return (prep, total - prep, total)
    

def calc_k2(k, w):
    k1 = k // 2
    k2 = k - k1
    prep, online, cost = cost_all([k1, k2], w)
    best = ((prep, online, cost), k1, k2, 0)
    print(f"k {k}: ({best[1]},{best[2]}) cost (prep, online, total): {best[0]} bits")

def calc_k3(k, w):
    k3_start = k//3
    best = None

    for k3 in range(k3_start, -1, -1):
        remaining = k - k3
        k2 = remaining // 2
        k1 = remaining - k2
        prep, online, cost = cost_all([k1, k2, k3], w)
        if best is None or cost < best[0][2]:
            best = ((prep, online, cost), k1, k2, k3, 0)
    if best is not None:
        print(f"k {k}: ({best[1]},{best[2]},{best[3]}) cost (prep, online, total): {best[0]} bits")

def calc_k4(k, w):
    best = None
    for k4 in range(k,-1,-1):
        rem = k - k4    
        for k3 in range(rem, -1, -1):
            remaining = rem - k3
            k2 = remaining // 2
            k1 = remaining - k2
            prep, online, cost = cost_all([k1, k2, k3, k4], w)
            if best is None or cost < best[0][2]:
                best = ((prep, online, cost), k1, k2, k3, k4)
    if best is not None:
        print(f"k {k}: ({best[1]},{best[2]},{best[3]},{best[4]}) cost (prep, online, total): {best[0]} bits")

def calc_k5(k, w):
    best = None
    for k5 in range(k,-1,-1):
        rem5 = k - k5
        for k4 in range(rem5,-1,-1):
            rem = rem5 - k4    
            for k3 in range(rem, -1, -1):
                remaining = rem - k3
                k2 = remaining // 2
                k1 = remaining - k2
                prep, online, cost = cost_all([k1, k2, k3, k4, k5], w)
                if best is None or cost < best[0][2]:
                    best = ((prep, online, cost), k1, k2, k3, k4, k5)
    
    if best is not None:
        print(f"k {k}: ({best[1]},{best[2]},{best[3]},{best[4]},{best[5]}) cost (prep, online, total): {best[0]} bits")


    
def main():
    parser = argparse.ArgumentParser(
        description="Compute optimal (k1,k2,k3) split for given k and weight w"
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,
        help="Weight parameter w used in the cost function",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=3,
        help="Dimensionality (default: 3)",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=1,
        help="Minimum k to iterate from (default: 1)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=24,
        help="Maximum k to iterate up to (default: 24)",
    )

    args = parser.parse_args()
    if args.w is None:
        print("Running with w set to k")

    for k in range(args.k_min, args.k_max + 1):
        w = args.w if args.w is not None else k
        k1 = k // 2
        k2 = k - k1
        baesline = 2**k1 + 2**k2 - k - 2
        match args.d:
            case 1:
                print(f"k {k} cost (prep, online, total): ({2**k - k - 1},{k},{2**k-1}) bits")
            case 2:
                calc_k2(k,w)
            case 3:
                calc_k3(k,w)
            case 4:
                calc_k4(k,w)
            case 5:
                calc_k5(k,w)
            case _:
                print("unsupported dimension")
        # print(f"\nk {k}: baseline cost: {baesline}\n")


if __name__ == "__main__":
    main()