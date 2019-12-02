cbuffer PerFrameCB
{
    int frame_number;
	int screen_width;
	int screen_height;
	int horizental_blocks_count;
};

Texture2D<float4> gCurPos; //world position
Texture2D<float4> gCurNorm; //world normal
Texture2D<float4> albedo;

RWTexture2D<float> tmp_data;// pixelCount * (FEATURES_COUNT + color_channels)  
RWTexture2D<float> out_data;// scaled features and filtered colors
RWTexture2D<float4> gCurNoisy; //current noisy image

groupshared float sum_vec[256];
groupshared float uVec[1024];
groupshared float rmat[10][13]; // FEATURES * BUFFER_COUNT
groupshared float u_length_squared;
groupshared float dotV;
groupshared float block_min;
groupshared float block_max;
groupshared float vec_length;

#define BUFFER_COUNT 13
#define FEATURES_COUNT 10
#define FEATURES_NOT_SCALED 4
#define BLOCK_PIXELS 1024
#define LOCAL_SIZE 256
#define BLOCK_EDGE_LENGTH 32
#define NOISE_AMOUNT 0.01
#define BLOCK_OFFSETS_COUNT 16

#define INBLOCK_ID sub_vector * LOCAL_SIZE + groupThreadId.x
#define BLOCK_OFFSET groupId.x * BUFFER_COUNT

static const int2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] =
{
	int2(-30, -30),
	int2(-12, -22),
	int2(-24, -2),
	int2(-8, -16),
	int2(-26, -24),
	int2(-14, -4),
	int2(-4, -28),
	int2(-26, -16),
	int2(-4, -2),
	int2(-24, -32),
	int2(-10, -10),
	int2(-18, -18),
	int2(-12, -30),
	int2(-32, -4),
	int2(-2, -20),
	int2(-22, -12),
};

// Simple mirroring of image index if it is out of bounds.
// NOTE: Works only if index is less than one size out of bounds.
static inline int mirror(int index, int size)
{
	if (index < 0)
		index = abs(index) - 1;
	else if (index >= size)
		index = 2 * size - index - 1;

	return index;
}

static inline int2 mirror2(int2 index, int2 size)
{
	index.x = mirror(index.x, size.x);
	index.y = mirror(index.y, size.y);

	return index;
}

static inline float random(uint a) {
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);

   return float(a) / 4294967296.0;
}

static inline float add_random(
      const float value,
      const int id,
      const int sub_vector,
      const int feature_buffer,
      const int frame_number){
   return value + NOISE_AMOUNT * 2 * (random(id + sub_vector * LOCAL_SIZE +
      feature_buffer * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH +
      frame_number * BUFFER_COUNT * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH) - 0.5f);
}

[numthreads(256, 1, 1)] // LOCAL_SIZE
void fit(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadId)
{
	// load features and colors to tmp_data
	for (int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
		int index = INBLOCK_ID;
		int2 uv = int2(groupId.x % horizental_blocks_count, groupId.x / horizental_blocks_count);
		uv *= BLOCK_EDGE_LENGTH;
		uv += int2(index % BLOCK_EDGE_LENGTH, index / BLOCK_EDGE_LENGTH);
		uv += BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT];
		uv = mirror2(uv, int2(screen_width, screen_height));
		tmp_data[uint2(index, 0 + BLOCK_OFFSET)] = 1;
        tmp_data[uint2(index, 1 + BLOCK_OFFSET)] = gCurNorm[uv].x;
        tmp_data[uint2(index, 2 + BLOCK_OFFSET)] = gCurNorm[uv].y;
        tmp_data[uint2(index, 3 + BLOCK_OFFSET)] = gCurNorm[uv].z;
        tmp_data[uint2(index, 4 + BLOCK_OFFSET)] = gCurPos[uv].x;
        tmp_data[uint2(index, 5 + BLOCK_OFFSET)] = gCurPos[uv].y;
        tmp_data[uint2(index, 6 + BLOCK_OFFSET)] = gCurPos[uv].z;
        tmp_data[uint2(index, 7 + BLOCK_OFFSET)] = gCurPos[uv].x * gCurPos[uv].x;
        tmp_data[uint2(index, 8 + BLOCK_OFFSET)] = gCurPos[uv].y * gCurPos[uv].y;
        tmp_data[uint2(index, 9 + BLOCK_OFFSET)] = gCurPos[uv].z * gCurPos[uv].z;
		tmp_data[uint2(index, 10 + BLOCK_OFFSET)] = gCurNoisy[uv].x / (albedo[uv].x + 0.1);
		tmp_data[uint2(index, 11 + BLOCK_OFFSET)] = gCurNoisy[uv].y / (albedo[uv].y + 0.1);
		tmp_data[uint2(index, 12 + BLOCK_OFFSET)] = gCurNoisy[uv].z / (albedo[uv].z + 0.1);
	}
	GroupMemoryBarrierWithGroupSync();

    for(int feature_buffer = FEATURES_NOT_SCALED; feature_buffer < FEATURES_COUNT; ++feature_buffer) {
        int sub_vector = 0;
        float tmp_max = tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)];
        float tmp_min = tmp_max;
        for(++sub_vector; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
            float value = tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)];
            tmp_max = max(value, tmp_max);
            tmp_min = min(value, tmp_min);
        }
        sum_vec[groupThreadId.x] = tmp_max;
        GroupMemoryBarrierWithGroupSync();
        
        // parallel reduction find max
        if(groupThreadId.x < 128) sum_vec[groupThreadId.x] = max(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 128]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 64) sum_vec[groupThreadId.x] = max(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 64]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 32) sum_vec[groupThreadId.x] = max(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 32]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 16) sum_vec[groupThreadId.x] = max(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 16]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 8) sum_vec[groupThreadId.x] = max(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 8]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 4) sum_vec[groupThreadId.x] = max(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 4]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 2) sum_vec[groupThreadId.x] = max(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 2]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x == 0) block_max = max(sum_vec[0], sum_vec[1]);
        GroupMemoryBarrierWithGroupSync();

        sum_vec[groupThreadId.x] = tmp_min;
        GroupMemoryBarrierWithGroupSync();
        
        // parallel reduction find min
        if(groupThreadId.x < 128) sum_vec[groupThreadId.x] = min(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 128]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 64) sum_vec[groupThreadId.x] = min(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 64]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 32) sum_vec[groupThreadId.x] = min(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 32]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 16) sum_vec[groupThreadId.x] = min(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 16]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 8) sum_vec[groupThreadId.x] = min(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 8]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 4) sum_vec[groupThreadId.x] = min(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 4]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 2) sum_vec[groupThreadId.x] = min(sum_vec[groupThreadId.x], sum_vec[groupThreadId.x + 2]);
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x == 0) block_min = min(sum_vec[0], sum_vec[1]);
		GroupMemoryBarrierWithGroupSync();

        // normalize feature
        if(block_max - block_min > 1) {
            for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
				out_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] = (tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] - block_min) / (block_max - block_min);
                tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] = out_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)];
            }
        } else {
            for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
                out_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] = tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] - block_min;
                tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] = out_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)];
            }
        }
    }

    // copy noise colors to out
    for(int feature_buffer = FEATURES_COUNT; feature_buffer < BUFFER_COUNT; ++feature_buffer) {
        for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
            out_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] = tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)];
        }
    }
    // copy not scaled features to out
    for(int feature_buffer = 0; feature_buffer < FEATURES_NOT_SCALED; ++feature_buffer) {
        for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
            out_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)] = tmp_data[uint2(INBLOCK_ID, feature_buffer + BLOCK_OFFSET)];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Householder QR decomposition
    for(int col = 0; col < FEATURES_COUNT; col++) {
        float tmp_sum_value = 0;
        for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
			int index = INBLOCK_ID;
            float tmp = out_data[uint2(index, col + BLOCK_OFFSET)];
            uVec[index] = tmp;
            if(index >= col + 1) {
                tmp_sum_value += tmp * tmp;
            }
        }
        sum_vec[groupThreadId.x] = tmp_sum_value;
        GroupMemoryBarrierWithGroupSync();

        // parallel reduction sum
        if(groupThreadId.x < 128) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 128];
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 64) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 64];
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 32) sum_vec[groupThreadId.x] +=sum_vec[groupThreadId.x + 32];
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 16) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 16];
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 8) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 8];
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 4) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 4];
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 2) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 2];
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x == 0) vec_length = sum_vec[0] + sum_vec[1];
        GroupMemoryBarrierWithGroupSync();
   
        float r_value;
        if(groupThreadId.x < col) {
            r_value = uVec[groupThreadId.x];
        } else if(groupThreadId.x == col) {
			u_length_squared = vec_length;
			vec_length = sqrt(vec_length + uVec[col] * uVec[col]);
			uVec[col] -= vec_length;
			u_length_squared += uVec[col] * uVec[col];
			r_value = vec_length;
        } else if(groupThreadId.x > col) { 
            r_value = 0;
        }

        if(groupThreadId.x < FEATURES_COUNT)
            rmat[groupThreadId.x][col] = r_value;

        for(int feature_buffer = col + 1; feature_buffer < BUFFER_COUNT; feature_buffer++) {
            float tmp_data_private_cache[BLOCK_PIXELS / LOCAL_SIZE];
            float tmp_sum_value = 0;
            for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
                int index = INBLOCK_ID;
                if(index >= col) {
                    float tmp = out_data[uint2(index, feature_buffer + BLOCK_OFFSET)];
                    if(col == 0 && feature_buffer < FEATURES_COUNT) {
                        tmp = add_random(tmp, groupThreadId.x, sub_vector, feature_buffer, frame_number);
                    }
                    tmp_data_private_cache[sub_vector] = tmp;
                    tmp_sum_value += tmp * uVec[index];
                }
            }

            sum_vec[groupThreadId.x] = tmp_sum_value;
            GroupMemoryBarrierWithGroupSync();
            // parallel reduction sum
            if(groupThreadId.x < 128) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 128];
            GroupMemoryBarrierWithGroupSync();
            if(groupThreadId.x < 64) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 64];
            GroupMemoryBarrierWithGroupSync();
            if(groupThreadId.x < 32) sum_vec[groupThreadId.x] +=sum_vec[groupThreadId.x + 32];
            GroupMemoryBarrierWithGroupSync();
            if(groupThreadId.x < 16) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 16];
            GroupMemoryBarrierWithGroupSync();
            if(groupThreadId.x < 8) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 8];
            GroupMemoryBarrierWithGroupSync();
            if(groupThreadId.x < 4) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 4];
            GroupMemoryBarrierWithGroupSync();
            if(groupThreadId.x < 2) sum_vec[groupThreadId.x] += sum_vec[groupThreadId.x + 2];
            GroupMemoryBarrierWithGroupSync();
            if(groupThreadId.x == 0) dotV = sum_vec[0] + sum_vec[1];
            GroupMemoryBarrierWithGroupSync();

            for (int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
                int index = INBLOCK_ID;
                if (index >= col) {
                    out_data[uint2(index, feature_buffer + BLOCK_OFFSET)] = tmp_data_private_cache[sub_vector]
                                                                - 2 * uVec[index] * dotV / u_length_squared;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    uint tmpId;
    if(groupThreadId.x < FEATURES_COUNT) {
        rmat[groupThreadId.x][FEATURES_COUNT] = out_data[uint2(groupThreadId.x, FEATURES_COUNT + BLOCK_OFFSET)];
    } else if((tmpId = groupThreadId.x - FEATURES_COUNT) < FEATURES_COUNT) {
        rmat[tmpId][BUFFER_COUNT - 2] = out_data[uint2(tmpId, BUFFER_COUNT - 2 + BLOCK_OFFSET)];
    } else if((tmpId = tmpId - FEATURES_COUNT) < FEATURES_COUNT) {
        rmat[tmpId][BUFFER_COUNT - 1] = out_data[uint2(tmpId, BUFFER_COUNT - 1 + BLOCK_OFFSET)];
    }
    GroupMemoryBarrierWithGroupSync();

    
    // Back substitution
    for(int i = BUFFER_COUNT - 4; i >= 0; i--) {
        if(groupThreadId.x < 3) {
			rmat[i][BUFFER_COUNT - groupThreadId.x - 1] /=  rmat[i][i];
        }
        GroupMemoryBarrierWithGroupSync();
        if(groupThreadId.x < 3 * i) {
            uint rowId = i - groupThreadId.x / 3 - 1;
            uint channel = BUFFER_COUNT - (groupThreadId.x % 3) - 1;
            rmat[rowId][channel] -= rmat[i][channel] * rmat[rowId][i];
        }
        GroupMemoryBarrierWithGroupSync();
    }
	
    // calculate filtered color
    for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
        int index = INBLOCK_ID;
		int2 uv = int2(groupId.x % horizental_blocks_count, groupId.x / horizental_blocks_count);
		uv *= BLOCK_EDGE_LENGTH;
		uv += int2(index % BLOCK_EDGE_LENGTH, index / BLOCK_EDGE_LENGTH);
		uv += BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT];
		if (uv.x < 0 || uv.y < 0 || uv.x >= screen_width || uv.y >= screen_height) {
			continue;
		}

		float rchannal = 0;
        for(int col = 0; col < FEATURES_COUNT; col++) {
			rchannal += rmat[col][FEATURES_COUNT] * tmp_data[uint2(index, col + BLOCK_OFFSET)];
        }
		rchannal = rchannal < 0 ? 0 : rchannal;

		float gchannal = 0;
        for(int col = 0; col < FEATURES_COUNT; col++) {
			gchannal += rmat[col][FEATURES_COUNT + 1] * tmp_data[uint2(index, col + BLOCK_OFFSET)];
        }
		gchannal = gchannal < 0 ? 0 : gchannal;

		float bchannal = 0;
        for(int col = 0; col < FEATURES_COUNT; col++) {
			bchannal += rmat[col][FEATURES_COUNT + 2] * tmp_data[uint2(index, col + BLOCK_OFFSET)];
        }
		bchannal = bchannal < 0 ? 0 : bchannal;
		gCurNoisy[uv] = (albedo[uv] + 0.1f) * float4(rchannal, gchannal, bchannal, gCurNoisy[uv].w);
	}
}