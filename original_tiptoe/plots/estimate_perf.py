import numpy
import math

WEB_NUM_DOCS = 437842961
WEB_NUM_LOGICAL_SERVERS = 88
WEB_SHARDS_PER_MACHINE = 2

CORES_PER_SERVER = 4
CORES_PER_COORDINATOR = 32 

WEB_EMBEDDING_L = 62499 
WEB_EMBEDDING_M = 1356096 #1355328
WEB_EMBEDDING_P = 131072
WEB_EMBEDDING_N = 2048.0 #1408.0
WEB_URL_L = 25125
WEB_URL_M = 507595 #507678
WEB_URL_P = 256
WEB_URL_N = 1408.0 #1024.0

WEB_EMBEDDING_DIM = 192
WEB_EMBEDDING_PREC = 5

WEB_HINT_SZ_MB = 7.16 #884.71
WEB_PCA_SZ_MB = 0.6
WEB_MODEL_SZ_MB = 265
WEB_CENTROIDS_SZ_MB = 68

IMAGE_CENTROIDS_SZ = 75

lwe = {}
lwe[32] = {}
lwe[32][1408] = {}
lwe[32][1408][1 << 13] = 991
lwe[32][1408][1 << 14] = 833
lwe[32][1408][1 << 15] = 701
lwe[32][1408][1 << 16] = 589
lwe[32][1408][1 << 17] = 495
lwe[32][1408][1 << 18] = 416
lwe[32][1408][1 << 19] = 350
lwe[32][1408][1 << 20] = 294
#lwe[32][1024][1 << 21] = 247 -- Commented out values are smaller than we need P to be :(

# NOTE! These will have a different std deviataion...
lwe[32][1608] = {}
lwe[32][1608][1 << 21] = 887
lwe[32][1608][1 << 22] = 745
lwe[32][1608][1 << 23] = 627
lwe[32][1608][1 << 24] = 527

lwe[64] = {}
lwe[64][2048] = {}
lwe[64][2048][1 << 13] = 1 << (math.floor(math.log2(574457)))
lwe[64][2048][1 << 14] = 1 << (math.floor(math.log2(483058)))
lwe[64][2048][1 << 15] = 1 << (math.floor(math.log2(406202)))
lwe[64][2048][1 << 16] = 1 << (math.floor(math.log2(341574)))
lwe[64][2048][1 << 17] = 1 << (math.floor(math.log2(287228)))
lwe[64][2048][1 << 18] = 1 << (math.floor(math.log2(241529)))
lwe[64][2048][1 << 19] = 1 << (math.floor(math.log2(203101)))
lwe[64][2048][1 << 20] = 1 << (math.floor(math.log2(170787)))
lwe[64][2048][1 << 21] = 1 << (math.floor(math.log2(143614)))
#lwe[64][1408][1 << 22] = 1 << (math.floor(math.log2(120764)))
#lwe[64][1408][1 << 23] = 1 << (math.floor(math.log2(101550)))
lwe[64][2048][1 << 22] = 1 << (math.floor(math.log2(540076)))
lwe[64][2048][1 << 23] = 1 << (math.floor(math.log2(454148)))
lwe[64][2048][1 << 24] = 1 << (math.floor(math.log2(381891)))

# Conclusion: The max dimension is 2048. So the RLWE portion will always have the same parameters!

def web_fixed_hint_size():
    return WEB_PCA_SZ_MB + WEB_MODEL_SZ_MB + WEB_CENTROIDS_SZ_MB

# warning: this is very conservative -- asumes whole coordinator is blocked
def online_tput_to_core_sec(tput, num_service_servers):
    return 1.0/tput * (num_service_servers * CORES_PER_SERVER / WEB_SHARDS_PER_MACHINE + CORES_PER_COORDINATOR) # accounts for number of cores for web search! 

def offline_tput_to_core_sec(tput):
    return 1.0/tput * CORES_PER_COORDINATOR # accounts for number of cores for web search! 

def web_inner_prod_range():
    return max_inner_prod(WEB_EMBEDDING_PREC, WEB_EMBEDDING_DIM)

def inner_prod_range(slot_bits, embedding_dim):
    return embedding_dim * (1 << (slot_bits-1)) * (1 << (slot_bits-1)) * 2

def scale_db_dims(scale):
    #scaleL = int(math.ceil(math.sqrt(float(scale))))
    #scaleM = int(math.ceil(float(scale) / scaleL))
    return math.sqrt(float(scale)), math.sqrt(float(scale))#scaleL, scaleM

def get_params_mod32(M):
    # For URL service, need to fit 1 byte in each entry
    logM = int(math.ceil(math.log2(M)))

    if logM <= 20:
        if logM <= 13:
            return 1408, lwe[32][1408][1 << 13]
        else:
            return 1408, lwe[32][1408][1 << logM]

    if logM <= 24:
        return 1608, lwe[32][1608][1 << logM]

    assert(False)

def get_params_mod64(M):
    # For embedding service, need to fit web_inner_prod_range() = 98305 bits in each entry
    logM = int(math.ceil(math.log2(M)))

    if logM <= 24:
        if logM <= 13:
            return 2048, lwe[64][2048][1 << 13]
        else:
            return 2048, lwe[64][2048][1 << logM]
   
    assert(False)

def extrapolate_offline_core_sec(measured_num_docs, measured_tput, scale):
    if measured_num_docs != WEB_NUM_DOCS:
        print("Not yet supported")
        assert(False)

    if scale == 0:
        return 0

    measured_core_sec = offline_tput_to_core_sec(measured_tput)

    scaleL, scaleM = scale_db_dims(scale)

    new_core_sec = measured_core_sec * scaleL
    # Warning: FOR THE URLS, THE SECURITY PARAM CHANGES

    return new_core_sec

def extrapolate_online_core_sec(measured_num_docs, measured_tput1, measured_tput2, num_servers1, num_servers2, scale):
    if measured_num_docs != WEB_NUM_DOCS:
        print("Not yet supported")
        assert(False)

    if scale == 0:
        return 0

    measured_core_sec1 = online_tput_to_core_sec(measured_tput1, num_servers1)
    measured_core_sec2 = online_tput_to_core_sec(measured_tput2, num_servers2)

    scaleL, scaleM = scale_db_dims(scale)

    new_core_sec1 = measured_core_sec1 * scale #scaleL * scaleM 
    new_core_sec2 = measured_core_sec2 * scale #scaleL * scaleM

    print("Extrapolate at ", scale, " (", measured_num_docs * scale, " docs): ", new_core_sec1 + new_core_sec2)

    return new_core_sec1 + new_core_sec2

def extrapolate_offline_comm(measured_num_docs, q, a, scale):
    if measured_num_docs != WEB_NUM_DOCS:
        print("Not yet supported")
        assert(False)

    if scale == 0:
        return q

    scaleL, scaleM = scale_db_dims(scale)

    new_q = q
    new_a = a * scaleL

    return new_q + new_a

def extrapolate_online_comm(measured_num_docs, q1, a1, q2, a2, scale):
    if measured_num_docs != WEB_NUM_DOCS:
        print("Not yet supported")
        assert(False)

    if scale == 0:
        return 0

    scaleL, scaleM = scale_db_dims(scale)

    new_q1 = q1 * scaleM
    new_q2 = q2 * scaleM
    new_a1 = a1 * scaleL
    new_a2 = a2 * scaleL

    return new_q1 + new_q2 + new_a1 + new_a2

