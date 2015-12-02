/*
 *  Basic CUDA based triangle mesh path tracer.
 *  For background info, see http://raytracey.blogspot.co.nz/2015/12/gpu-path-tracing-tutorial-2-interactive.html
 *  Based on CUDA ray tracing code from http://cg.alexandra.dk/?p=278
 *  Copyright (C) 2015  Sam Lapere 
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"
#include "cutil_math.h"  // required for float3 vector math
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\extras\CUPTI\include\GL\glew.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\extras\CUPTI\include\GL\glut.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#define M_PI 3.14159265359f
#define width 1024	// screenwidth
#define height 576	// screenheight
#define samps  1	// samples per pixel per pass

int total_number_of_triangles = 0;
int frames = 0;

// scene bounding box
float3 scene_aabbox_min;
float3 scene_aabbox_max;

// the scene triangles are stored in a 1D CUDA texture of float4 for memory alignment
// each triangle is stored as the 3 vertices after each other
texture<float4, 1, cudaReadModeElementType> triangle_texture;  

// hardcoded camera position
__device__ float3 firstcamorig = { 50, 52, 295.6 };

// OpenGL vertex buffer object for real-time viewport
GLuint vbo;
void *d_vbo_buffer = NULL;

struct Ray {
	float3 orig;	// ray origin
	float3 dir;		// ray direction	
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance(), only DIFF used here

// SPHERES

struct Sphere {

	float rad;				// radius 
	float3 pos, emi, col;	// position, emission, color 
	Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = pos - r.orig;  // 
		float t, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
	}
};

// TRIANGLES
// the classic ray triangle intersection: http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
// for an explanation see http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection

__device__ float RayTriangleIntersection(const Ray &r,
	const float3 &v0,
	const float3 &edge1,
	const float3 &edge2)
{

	float3 tvec = r.orig - v0;
	float3 pvec = cross(r.dir, edge2);
	float  det = dot(edge1, pvec);

	det = __fdividef(1.0f, det);  // CUDA intrinsic function 

	float u = dot(tvec, pvec) * det;

	if (u < 0.0f || u > 1.0f)
		return -1.0f;

	float3 qvec = cross(tvec, edge1);

	float v = dot(r.dir, qvec) * det;

	if (v < 0.0f || (u + v) > 1.0f)
		return -1.0f;

	return dot(edge2, qvec) * det;
}

__device__ float3 getTriangleNormal(const int triangleIndex){

	float4 edge1 = tex1Dfetch(triangle_texture, triangleIndex * 3 + 1);
	float4 edge2 = tex1Dfetch(triangle_texture, triangleIndex * 3 + 2);

	float3 trinormal = cross(make_float3(edge1.x, edge1.y, edge1.z), make_float3(edge2.x, edge2.y, edge2.z));
	trinormal = normalize(trinormal);

	return trinormal;
}

__device__ void intersectAllTriangles(const Ray& r, float& t_scene, int& triangle_id, const int number_of_triangles, int& geomtype){

	for (int i = 0; i < number_of_triangles; i++)
	{
	// the triangles are packed into the 1D texture in a way that each float4 contains 
	// either the first triangle-vertex or two triangle edges, like this: 
	// (float4(vertex.x,vertex.y,vertex.z, 0), float4 (egde1.x,egde1.y,egde1.z,0),float4 (egde2.x,egde2.y,egde2.z,0)) for each triangle.
		
		float4 v0 = tex1Dfetch(triangle_texture, i * 3);
		float4 edge1 = tex1Dfetch(triangle_texture, i * 3 + 1);
		float4 edge2 = tex1Dfetch(triangle_texture, i * 3 + 2);

		float t = RayTriangleIntersection(r,
			make_float3(v0.x, v0.y, v0.z),
			make_float3(edge1.x, edge1.y, edge1.z),
			make_float3(edge2.x, edge2.y, edge2.z));

		if (t < t_scene && t > 0.001)
		{
			t_scene = t;
			triangle_id = i;
			geomtype = 3;
		}
	}
}


// AXIS ALIGNED BOXES
// helper functions
inline __device__ float3 minf3(float3 a, float3 b){ return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b){ return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b){ return a < b ? a : b; }
inline __device__ float maxf1(float a, float b){ return a > b ? a : b; }

struct Box {

	float3 min;
	float3 max;
	float3 emi;
	float3 col;
	Refl_t refl;

	__device__ float intersect(const Ray &r) const {

		float3 tmin = (min - r.orig) / r.dir;
		float3 tmax = (max - r.orig) / r.dir;

		float epsilon = 0.001f;

		float3 real_min = minf3(tmin, tmax);
		float3 real_max = maxf3(tmin, tmax);

		float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
		float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

		if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
		else return 0;	
	}

	// calculate normal for point on axis aligned box
	__device__ float3 Box::normalAt(float3 &point) {

		float3 normal = make_float3(0.f, 0.f, 0.f);
		float min_distance = 1e8;
		float distance;
		float epsilon = 0.001f;

		if (fabs(min.x - point.x) < epsilon) normal = make_float3(-1, 0, 0);
		else if (fabs(max.x - point.x) < epsilon) normal = make_float3(1, 0, 0);
		else if (fabs(min.y - point.y) < epsilon) normal = make_float3(0, -1, 0);
		else if (fabs(max.y - point.y) < epsilon) normal = make_float3(0, 1, 0);
		else if (fabs(min.z - point.z) < epsilon) normal = make_float3(0, 0, -1);
		else normal = make_float3(0, 0, 1);

		return normal;
	}
};

// scene: 9 spheres forming a Cornell box
// small enough to fit in constant GPU memory
__constant__ Sphere spheres[] = {
	// FORMAT: { float radius, float3 position, float3 emission, float3 colour, Refl_t material }
	// cornell box
	//{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 1e5f
	//{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right 
	//{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
	//{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 0.00f, 0.00f, 0.00f }, DIFF }, //Front 
	//{ 1e5f, { 50.0f, -1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Bottom 
	//{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
	//{ 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 0.99f, 0.99f, 0.99f }, SPEC }, // small sphere 1
	//{ 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.f, .0f }, { 0.09f, 0.49f, 0.3f }, REFR }, // small sphere 2
	//{ 600.0f, { 50.0f, 681.6f - .5f, 81.6f }, { 3.0f, 2.5f, 2.0f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light 12, 10 ,8

	//outdoor scene: radius, position, emission, color, material

	//{ 1600, { 3000.0f, 10, 6000 }, { 37, 34, 30 }, { 0.f, 0.f, 0.f }, DIFF },  // 37, 34, 30 // sun
	//{ 1560, { 3500.0f, 0, 7000 }, { 50, 25, 2.5 }, { 0.f, 0.f, 0.f }, DIFF },  //  150, 75, 7.5 // sun 2
	{ 10000, { 50.0f, 40.8f, -1060 }, { 0.0003, 0.01, 0.15 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // sky
	{ 100000, { 50.0f, -100000, 0 }, { 0.0, 0.0, 0 }, { 0.8f, 0.2f, 0.f }, DIFF }, // ground
	{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, DIFF },  // horizon brightener
	{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF }, // mountains
	{ 82.5, { 30.0f, 180.5, 42 }, { 16, 12, 6 }, { .6f, .6f, 0.6f }, DIFF },  // small sphere 1
	{ 12, { 115.0f, 10, 105 }, { 0.0, 0.0, 0.0 }, { 0.9f, 0.9f, 0.9f }, REFR },  // small sphere 2
	{ 22, { 65.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, SPEC }, // small sphere 3
};

__constant__ Box boxes[] = {
// FORMAT: { float3 minbounds,    float3 maxbounds,         float3 emission,    float3 colour,       Refl_t }
	{ { 5.0f, 0.0f, 70.0f }, { 45.0f, 11.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
	{ {85.0f, 0.0f, 95.0f }, { 95.0f, 20.0f, 105.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
	{ {75.0f, 20.0f, 85.0f}, { 105.0f, 22.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
};


__device__ inline bool intersect_scene(const Ray &r, float &t, int &sphere_id, int &box_id, int& triangle_id, const int number_of_triangles, int &geomtype, const float3& bbmin, const float3& bbmax){

	float tmin = 1e20;
	float tmax = -1e20;
	float d = 1e21;
	float k = 1e21;
	float q = 1e21;
	float inf = t = 1e20;

	// SPHERES
	// intersect all spheres in the scene
	float numspheres = sizeof(spheres) / sizeof(Sphere);
	for (int i = int(numspheres); i--;)  // for all spheres in scene
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(r)) && d < t){ t = d; sphere_id = i; geomtype = 1; }

	// BOXES
	// intersect boxes in the scene
	float numboxes = sizeof(boxes) / sizeof(Box);
	for (int i = int(numboxes); i--;) // for all boxes in scene
		if ((k = boxes[i].intersect(r)) && k < t){ t = k; box_id = i; geomtype = 2; }

	// TRIANGLES
	Box scene_bbox; // bounding box around triangle meshes
	scene_bbox.min = bbmin;
	scene_bbox.max = bbmax;

	// if ray hits bounding box of triangle meshes, intersect ray with all triangles
	if (scene_bbox.intersect(r)){
		intersectAllTriangles(r, t, triangle_id, number_of_triangles, geomtype);
	}
	
	// t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
	return t<inf;
}


// hash function to calculate new seed for each frame
uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray &r, curandState *randstate, const int totaltris, const float3& scene_aabb_min, const float3& scene_aabb_max){ // returns ray color

	// colour mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);

	for (int bounces = 0; bounces < 5; bounces++){  // iteration up to 4 bounces (instead of recursion in CPU code)

		// reset scene intersection function parameters
		float t = 100000; // distance to intersection 
		int sphere_id = -1;
		int box_id = -1;   // index of intersected sphere 
		int triangle_id = -1;
		int geomtype = -1;
		float3 f;  // primitive colour
		float3 emit; // primitive emission colour
		float3 x; // intersection point
		float3 n; // normal
		float3 nl; // oriented normal
		float3 d; // ray direction of next path segment
		Refl_t refltype;

		// intersect ray with scene
		// intersect_scene keeps track of closest intersected primitive and distance to closest intersection point
		if (!intersect_scene(r, t, sphere_id, box_id, triangle_id, totaltris, geomtype, scene_aabb_min, scene_aabb_max))
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

		// else: we've got a hit with a scene primitive
		// determine geometry type of primitive: sphere/box/triangle

		// if sphere:
		if (geomtype == 1){
			Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
			x = r.orig + r.dir*t;  // intersection point on object
			n = normalize(x - sphere.pos);		// normal
			nl = dot(n, r.dir) < 0 ? n : n * -1; // correctly oriented normal
			f = sphere.col;
			refltype = sphere.refl;
			emit = sphere.emi;
			accucolor += (mask * emit);
		}

		// if box:
		if (geomtype == 2){
			Box &box = boxes[box_id];
			x = r.orig + r.dir*t;  // intersection point on object
			n = normalize(box.normalAt(x)); // normal
			nl = n;// dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
			f = box.col;
			refltype = box.refl;
			emit = box.emi;
			accucolor += (mask * emit);
		}

		// if triangle:
		if (geomtype == 3){
			int tri_index = triangle_id;
			x = r.orig + r.dir*t;  // intersection point
			n = normalize(getTriangleNormal(tri_index));  // normal 
			nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
			
			// colour, refltype and emit value are hardcoded and apply to all triangles
			// no per triangle material support yet
			f = make_float3(0.9f, 0.4f, 0.1f);  // triangle colour
			refltype = REFR;  
			emit = make_float3(0.0f, 0.0f, 0.0f); 
			accucolor += (mask * emit);
		}

		// SHADING: diffuse, specular or refractive
		
		// ideal diffuse reflection (see "Realistic Ray Tracing", P. Shirley)
		if (refltype == DIFF){ 

			// create 2 random numbers
			float r1 = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float3 w = nl;
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
			
			// offset origin next path segment to prevent self intersection
			x += nl * 0.03;

			// multiply mask with colour of object
			mask *= f;
		}

		// ideal specular reflection (mirror) 
		if (refltype == SPEC){  

			// compute relfected ray direction according to Snell's law
			d = r.dir - 2.0f * n * dot(n, r.dir);
			
			// offset origin next path segment to prevent self intersection
			x += nl * 0.01f;
		
			// multiply mask with colour of object
			mask *= f; 
		}

		// ideal refraction (based on smallpt code by Kevin Beason)
		if (refltype == REFR){ 

			bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(r.dir, nl); 
			float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				d = reflect(r.dir, n); //d = r.dir - 2.0f * n * dot(n, r.dir);
				x += nl * 0.01f;
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float3 tdir = normalize(r.dir * nnt - n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

				float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re; 
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.25) // reflect
				{
					mask *= RP;
					d = reflect(r.dir, n);
					x += nl * 0.02f;
				}
				else // transmit
				{
					mask *= TP;
					d = tdir; //r = Ray(x, tdir); 
					x += nl * 0.0005f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		r.orig = x; 
		r.dir = d;
	}

	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return accucolor;
}

// required to convert colour to a format that OpenGL can display  
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__global__ void render_kernel(float3 *output, float3* accumbuffer, const int numtriangles, int framenumber, uint hashedframenumber, float3 scene_bbmin, float3 scene_bbmax){   // float3 *gputexdata1, int *texoffsets

	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	Ray cam(firstcamorig, normalize(make_float3(0, -0.042612, -1)));
	float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f);  // ray direction offset along X-axis 
	float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray dir offset along Y-axis, .5135 is FOV angle
	float3 pixelcol; // final pixel color       

	int i = (height - y - 1)*width + x; // pixel index

	pixelcol = make_float3(0.0f, 0.0f, 0.0f); // reset to zero for every pixel	

	for (int s = 0; s < samps; s++){ 

		// compute primary ray direction
		float3 d = cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5) + cam.dir;
		// normalize primary ray direction
		d = normalize(d);
		// add accumulated colour from path bounces
		pixelcol += radiance(Ray(cam.orig + d * 40, d), &randState, numtriangles, scene_bbmin, scene_bbmax)*(1. / samps);   
	}       // Camera rays are pushed ^^^^^ forward to start in interior 

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += pixelcol;
	// averaged colour: divide colour by the number of calculated frames so far
	float3 tempcol = accumbuffer[i] / framenumber;

	Colour fcolour;
	float3 colour = make_float3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f)); 
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = make_float3(x, y, fcolour.c);
}

void Timer(int obsolete) {

	glutPostRedisplay();
	glutTimerFunc(30, Timer, 0);
}

void createVBO(GLuint* vbo)
{
	//create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//initialize VBO
	unsigned int size = width * height * sizeof(float3);  // 3 floats
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

__device__ float timer = 0.0f;

inline float clamp(float x){ return x<0 ? 0 : x>1 ? 1 : x; }

//inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  // RGB float in range [0,1] to int in range [0, 255]

// initialise OpenGL viewport
void init()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
}

// buffer for accumulating samples over several frames
float3* accumulatebuffer;
// output buffer
float3 *dptr;

void disp(void)  
{
	frames++;
	cudaThreadSynchronize();

	// map vertex buffer object for acces by CUDA 
	cudaGLMapBufferObject((void**)&dptr, vbo);

	//clear all pixels:
	glClear(GL_COLOR_BUFFER_BIT);

	// RAY TRACING:
	// dim3 grid(WINDOW / block.x, WINDOW / block.y, 1);
	// dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 block(16, 16, 1);   
	dim3 grid(width / block.x, height / block.y, 1);

	// launch CUDA path tracing kernel, pass in a hashed seed based on number of frames
	render_kernel <<< grid, block >>>(dptr, accumulatebuffer, total_number_of_triangles, frames, WangHash(frames), scene_aabbox_max, scene_aabbox_min);  // launches CUDA render kernel from the host

	cudaThreadSynchronize();

	// unmap buffer
	cudaGLUnmapBufferObject(vbo);
	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	//glutPostRedisplay();
}

// load triangle data in a CUDA texture
extern "C"
{
	void bindTriangles(float *dev_triangle_p, unsigned int number_of_triangles)
	{
		triangle_texture.normalized = false;                      // access with normalized texture coordinates
		triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
		triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4)*number_of_triangles * 3;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(0, triangle_texture, dev_triangle_p, channelDesc, size);
	}
}

// helpers to load triangle data
struct TriangleFace
{
	int v[3]; // vertex indices
};

struct TriangleMesh
{
	std::vector<float3> verts;
	std::vector<TriangleFace> faces;
	float3 bounding_box[2];
};

TriangleMesh mesh1;
TriangleMesh mesh2;

float *dev_triangle_p; // the cuda device pointer that points to the uploaded triangles

void loadObj(const std::string filename, TriangleMesh &mesh); // forward declaration

void initCUDAmemoryTriMesh()
{
	loadObj("data/teapot.obj", mesh1);
	loadObj("data/bunny.obj", mesh2);

	// scalefactor and offset to position/scale triangle meshes
	float scalefactor1 = 8;
	float scalefactor2 = 300;  // 300
	float3 offset1 = make_float3(90, 22, 100);// (30, -2, 80);
	float3 offset2 = make_float3(30, -2, 80);

	std::vector<float4> triangles;

	for (unsigned int i = 0; i < mesh1.faces.size(); i++)
	{
		// make a local copy of the triangle vertices
		float3 v0 = mesh1.verts[mesh1.faces[i].v[0] - 1];
		float3 v1 = mesh1.verts[mesh1.faces[i].v[1] - 1];
		float3 v2 = mesh1.verts[mesh1.faces[i].v[2] - 1];

		// scale
		v0 *= scalefactor1;
		v1 *= scalefactor1;
		v2 *= scalefactor1;

		// translate
		v0 += offset1;
		v1 += offset1;
		v2 += offset1;

		// store triangle data as float4
		// store edges instead of vertex points, to save some calculations in the
		// ray triangle intersection test
		triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
		triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));  
		triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0)); 
	}

	// compute bounding box of this mesh
	mesh1.bounding_box[0] *= scalefactor1; mesh1.bounding_box[0] += offset1;
	mesh1.bounding_box[1] *= scalefactor1; mesh1.bounding_box[1] += offset1;

	for (unsigned int i = 0; i < mesh2.faces.size(); i++)
	{
		float3 v0 = mesh2.verts[mesh2.faces[i].v[0] - 1];
		float3 v1 = mesh2.verts[mesh2.faces[i].v[1] - 1];
		float3 v2 = mesh2.verts[mesh2.faces[i].v[2] - 1];

		v0 *= scalefactor2;
		v1 *= scalefactor2;
		v2 *= scalefactor2;

		v0 += offset2;
		v1 += offset2;
		v2 += offset2;

		triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
		triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 1)); 
		triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0)); 
	}

	mesh2.bounding_box[0] *= scalefactor2; mesh2.bounding_box[0] += offset2;
	mesh2.bounding_box[1] *= scalefactor2; mesh2.bounding_box[1] += offset2;

	std::cout << "total number of triangles check:" << mesh1.faces.size() + mesh2.faces.size() << " == " << triangles.size() / 3 << std::endl;

	// calculate total number of triangles in the scene
	size_t triangle_size = triangles.size() * sizeof(float4);
	int total_num_triangles = triangles.size() / 3;
	total_number_of_triangles = total_num_triangles;

	if (triangle_size > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void **)&dev_triangle_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_triangle_p, &triangles[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bindTriangles(dev_triangle_p, total_num_triangles);
	}

	// compute scene bounding box by merging bounding boxes of individual meshes 
	scene_aabbox_min = mesh2.bounding_box[0];
	scene_aabbox_max = mesh2.bounding_box[1];
	scene_aabbox_min = fminf(scene_aabbox_min, mesh1.bounding_box[0]);
	scene_aabbox_max = fmaxf(scene_aabbox_max, mesh1.bounding_box[1]);

}

// read triangle data from obj file
void loadObj(const std::string filename, TriangleMesh &mesh)
{
	std::ifstream in(filename.c_str());

	if (!in.good())
	{
		std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
		system("PAUSE");
		exit(0);
	}

	char buffer[256], str[255];
	float f1, f2, f3;

	while (!in.getline(buffer, 255).eof())
	{
		buffer[255] = '\0';
		sscanf_s(buffer, "%s", str, 255);

		// reading a vertex
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)){
			if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3){
				mesh.verts.push_back(make_float3(f1, f2, f3));
			}
			else{
				std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}

		// reading faceMtls 
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			TriangleFace f;
			int nt = sscanf(buffer, "f %d %d %d", &f.v[0], &f.v[1], &f.v[2]);
			if (nt != 3){
				std::cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}

			mesh.faces.push_back(f);
		}
	}

	// calculate the bounding box of the mesh
	mesh.bounding_box[0] = make_float3(1000000, 1000000, 1000000);
	mesh.bounding_box[1] = make_float3(-1000000, -1000000, -1000000);
	for (unsigned int i = 0; i < mesh.verts.size(); i++)
	{
		//update min and max value
		mesh.bounding_box[0] = fminf(mesh.verts[i], mesh.bounding_box[0]);
		mesh.bounding_box[1] = fmaxf(mesh.verts[i], mesh.bounding_box[1]);
	}

	std::cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << std::endl;
	std::cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z << ") max:"
		<< mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z << ")" << std::endl;
}

int main(int argc, char** argv){

	// allocate memmory for the accumulation buffer on the GPU
	cudaMalloc(&accumulatebuffer, width * height * sizeof(float3));
	// load triangle meshes in CUDA memory
	initCUDAmemoryTriMesh();

	// init glut for OpenGL viewport
	glutInit(&argc, argv);
	// specify the display mode to be RGB and single buffering
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position
	glutInitWindowPosition(100, 100);
	// specify the initial window size
	glutInitWindowSize(width, height);
	// create the window and set title
	glutCreateWindow("Basic triangle mesh path tracer in CUDA");
	
	// init OpenGL
	init();
	fprintf(stderr, "OpenGL initialized \n");
	// register callback function to display graphics:
	glutDisplayFunc(disp);
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");
	// call Timer():
	Timer(0);
	createVBO(&vbo);
	fprintf(stderr, "VBO created  \n");
	// enter the main loop and process events
	fprintf(stderr, "Entering glutMainLoop...  \n");
	glutMainLoop();

	// free CUDA memory on exit
	cudaFree(accumulatebuffer);
	cudaFree(dev_triangle_p);
	cudaFree(dptr);
}
