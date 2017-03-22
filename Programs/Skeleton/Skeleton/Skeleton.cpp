//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ki kell kommentelni!
#include <iostream>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 ModelWorldScale;
	uniform mat4 ModelWorldRotation;
	uniform mat4 ModelWorldTranslation;
	uniform mat4 WorldView;

	layout(location = 0) in vec3 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vec3(0.0,1.0-vertexPosition.z/500.0,vertexPosition.z/500.0);
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, vertexPosition.z, 1)
						*ModelWorldScale
						*ModelWorldRotation
						*ModelWorldTranslation
						*WorldView;
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator*(const float& f) {
		return vec4(v[0]*f, v[1] * f, v[2] * f);
	}

	vec4 operator+=(const vec4 p) {
		return *this = *this + p;
	}
	vec4 operator+(const vec4 p) {
		return vec4(v[0] + p.v[0], v[1] + p.v[1], v[2] + p.v[2]);
	}

	vec4 normalize() {
		return *this*(1 / this->length());
	}

	float length() {
		return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	float& operator[](char c) {
		if (c == 'x') return v[0];
		if (c == 'y') return v[1];
		if (c == 'z') return v[2];
		//return v[index];
	}
};


// handle of the shader program
unsigned int shaderProgram;

// 3D camera
struct Camera3D {
	//float wCx, wCy;	// center in world coordinates
	vec4 position;
	vec4 scale;
	vec4 rotation;
	//float wWx, wWy;	// width and height in world coordinates
public:
	Camera3D() {
		Animate(0);
	}

	void Animate(float t) {

		position = vec4(0, 0, 0);
		scale = vec4(800,800,800);
		rotation = vec4(-90.0*(3.14 / 180) + 0.3, 0, t);
		//rotation = vec4(0,0,0);

		// commit uniform variables
		mat4 WorldViewRotationZ(
			cos(rotation['z']), -sin(rotation['z']), 0, 0,
			sin(rotation['z']), cos(rotation['z']), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);
		mat4 WorldViewRotationY(
			cos(rotation['y']), 0, sin(rotation['y']), 0,
			0, 1, 0, 0,
			-sin(rotation['y']), 0, cos(rotation['y']), 0,
			0, 0, 0, 1
		);
		mat4 WorldViewRotationX(
			1, 0, 0, 0,
			0, cos(rotation['x']), sin(rotation['x']), 0,
			0, -sin(rotation['x']), cos(rotation['x']), 0,
			0, 0, 0, 1
		);
		mat4 WorldViewTranslation(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0 - position['x'], 0 - position['y'], 0 - position['z'], 1
		);
		mat4 WorldViewScale(
			1/scale['x'], 0, 0, 0,
			0, 1 / scale['y'], 0, 0,
			0, 0, 1 / scale['z'], 0,
			0, 0, 0, 1
		);
		mat4 WorldView = WorldViewTranslation*WorldViewRotationZ*WorldViewRotationY*WorldViewRotationX*WorldViewScale;

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "WorldView");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, WorldView); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");
	}
};

// 2D camera
Camera3D* camera;



class Triangle3D {
	unsigned int vao;
public:
	Triangle3D(vec4 v1, vec4 v2, vec4 v3) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		float vertexCoords[] = { v1['x'], v1['y'], v1['z'], v2['x'], v2['y'], v2['z'],v3['x'], v3['y'], v3['z'] };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			3, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
		Animate(0);
	}

	void Animate(float t) {
		
	}

	void set(vec4 v1, vec4 v2, vec4 v3) {

	}

	void Draw() {

	//	//mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

		mat4 ModelWorldScale(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);

		mat4 ModelWorldRotation(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);

		mat4 ModelWorldTranslation(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "ModelWorldScale");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, ModelWorldScale); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		location = glGetUniformLocation(shaderProgram, "ModelWorldRotation");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, ModelWorldRotation); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		location = glGetUniformLocation(shaderProgram, "ModelWorldTranslation");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, ModelWorldTranslation); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_LINE_LOOP, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};


using namespace std;

class Point {

	vector<Triangle3D> triangles;
public:
	Point(vec4 p):triangles(vector<Triangle3D>()) {
		mat4 T = mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			p['x'], p['y'], p['z'], 1
		);

		mat4 S = mat4(
			10, 0, 0, 0,
			0, 10, 0, 0,
			0, 0, 10, 0,
			0, 0, 0, 1
		);

		mat4 M = S*T;
		triangles.push_back(Triangle3D(
			vec4(0, 0, 0)*M,
			vec4(1, 1, 2)*M,
			vec4(1, -1, 2)*M
		));
		triangles.push_back(Triangle3D(
			vec4(0, 0, 0)*M,
			vec4(1, -1, 2)*M,
			vec4(-1, -1, 2)*M
		));
		triangles.push_back(Triangle3D(
			vec4(0, 0, 0)*M,
			vec4(-1, -1, 2)*M,
			vec4(-1, 1, 2)*M
		));
		triangles.push_back(Triangle3D(
			vec4(0, 0, 0)*M,
			vec4(-1, 1, 2)*M,
			vec4(1, 1, 2)*M
		));
	}

	void draw() {
		for (size_t i = 0; i < triangles.size(); i++) {
			triangles[i].Draw();
		}
	}
};

class BezierSurface {
	vector<vector<vec4>> controlPoints;
	vector<vector<vec4>> interpolatedPoints;
	vector<Triangle3D> controlSurface;
	vector<Triangle3D> interpolatedSurface;
	vector<Triangle3D> grads;

	void createControlPoints(int n) {
		n = 5;
		float distanceUnit = 1000 / (n - 1); // világ 1km azaz 1000m széles és hosszú

		for (size_t i = 0; i < n; i++) {
			// a sor tárolójának létrehozása
			controlPoints.push_back(vector<vec4>());
			for (size_t j = 0; j < n; j++) {
				float x = (-500) + j * distanceUnit; // -500m és 500m közötti a világunk
				float y = (-500) + i * distanceUnit;
				float z = rand() % 500;
				controlPoints[i].push_back(vec4(x,y,z));
			}
		}
		
		controlPoints[0][0]['z'] = 0;
		controlPoints[0][1]['z'] = 300;
		controlPoints[0][2]['z'] = 300;
		controlPoints[0][3]['z'] = 0;
		controlPoints[0][4]['z'] = 0;

		controlPoints[1][0]['z'] = 500;
		controlPoints[1][1]['z'] = 0;
		controlPoints[1][2]['z'] = 300;
		controlPoints[1][3]['z'] = 300;
		controlPoints[1][4]['z'] = 0;

		controlPoints[2][0]['z'] = 500;
		controlPoints[2][1]['z'] = 0;
		controlPoints[2][2]['z'] = 0;
		controlPoints[2][3]['z'] = 300;
		controlPoints[2][4]['z'] = 0;

		controlPoints[3][0]['z'] = 0;
		controlPoints[3][1]['z'] = 300;
		controlPoints[3][2]['z'] = 300;
		controlPoints[3][3]['z'] = 500;
		controlPoints[3][4]['z'] = 500;

		controlPoints[4][0]['z'] = 0;
		controlPoints[4][1]['z'] = 0;
		controlPoints[4][2]['z'] = 0;
		controlPoints[4][3]['z'] = 500;
		controlPoints[4][4]['z'] = 500;
	}

	void createInterpolatedPoints(int m) {
		float distanceUnit = 1.0 / (m - 1);

		for (size_t i = 0; i < m; i++) {
			// a sor tárolójának létrehozása
			interpolatedPoints.push_back(vector<vec4>());
			for (size_t j = 0; j < m; j++) {
				float u = j * distanceUnit; // 0 - 1
				float v = i * distanceUnit;
				float z = getHeightAtPosition(u,v);
				//cout << "x: " << x << " y: " << y << " z: " << z << endl;
				interpolatedPoints[i].push_back(vec4(u*1000-500, v*1000-500, z));
			}
		}
	}


	float getHeightAtPosition(float u, float v) {
		float z = 0;
		for (size_t i = 0; i < controlPoints.size(); i++) {
			for (size_t j = 0; j < controlPoints[i].size(); j++) {
				float Wx = Weight(controlPoints[i].size() - 1, j, u);
				float Wy = Weight(controlPoints.size() - 1, i, v);
				z += Wx*Wy*controlPoints[i][j]['z'];
			}
		}
		return z;
	}

	float Weight(size_t n, size_t i, float u) {
		return ((float)factorial(n) / (factorial(i)*factorial(n - i)) * pow(u, i) * pow(1 - u, n - i));
	}

	// jó
	size_t factorial(size_t n) {
		if (n == 0) return 1;
		else return n*factorial(n - 1);
	}

	void drawSurface(vector<Triangle3D> surface) {
		for (size_t i = 0; i < surface.size(); i++) {
			surface[i].Draw();
		}
	}

	vector<Triangle3D> createSurface(vector<vector<vec4>> points) {
		vector<Triangle3D> surface;
		// sorok
		for (size_t i = 0; i < points.size() - 1; i++) {
			// pontok egy soron belül
			for (size_t j = 0; j < points[i].size() - 1; j++) {
				surface.push_back(Triangle3D(
					points[i][j],
					points[i][j + 1],
					points[i + 1][j]
				));
				surface.push_back(Triangle3D(
					points[i][j + 1],
					points[i + 1][j],
					points[i + 1][j + 1]
				));
			}
		}

		return surface;
	}

public:
	void createGrads() {
		size_t m = interpolatedPoints.size();
		float distanceUnit = 1.0 / (m - 1);

		for (size_t i = 0; i < m; i++) {
			// a sor tárolójának létrehozása
			//interpolatedPoints.push_back(vector<vec4>());
			for (size_t j = 0; j < m; j++) {
				float u = j * distanceUnit; // 0 - 1
				float v = i * distanceUnit;
				float z = getHeightAtPosition(u, v);
				//cout << "x: " << x << " y: " << y << " z: " << z << endl;
				//interpolatedPoints[i].push_back(vec4(u * 1000 - 500, v * 1000 - 500, z));

				grads.push_back(Triangle3D(
					vec4(u * 1000 - 500, v * 1000 - 500, z),
					vec4(u * 1000 - 510, v * 1000 - 510, z),// picit
															//vec4(u * 1000 - 510, v * 1000 - 510, z+30)// picit
					vec4(u * 1000 - 500, v * 1000 - 500, z) + getRollPosition(u, v)

				));
			}
		}
	}
	// default constructor
	BezierSurface() {
		createControlPoints(5);
		createInterpolatedPoints(11);
	}

	void draw() {
		//drawSurface(controlSurface);
		drawSurface(interpolatedSurface);
			/*for (size_t i = 0; i < grads.size(); i++) {
				grads[i].Draw();
			}*/
	}

	void createControlSurface() {
		controlSurface = createSurface(controlPoints);
	}

	void createInterpolatedSurface() {
		interpolatedSurface = createSurface(interpolatedPoints);
	}

	float derivatedWeight(size_t n, size_t i,float u) {
		return ((float)factorial(n) / (factorial(i)*factorial(n - i)) * (
			i*pow(u, i-1) * pow(1 - u, n - i)) + pow(u, i) * (n-i)* pow(1 - u, n - i-1)*(-1)
		);
	}

	vec4 getRollPosition(float u, float v) {//, vec4 e) {
		// ??????
		vec4 z;
		for (size_t i = 0; i < controlPoints.size(); i++) {
			for (size_t j = 0; j < controlPoints[i].size(); j++) {
				float Wx = derivatedWeight(controlPoints[i].size() - 1, j, u);// *e.normalize()['x'];
				float Wy = derivatedWeight(controlPoints.size() - 1, i, v);// *e.normalize()['y'];
				z += controlPoints[i][j]* Wx*Wy;
			}
		}
		return z;//atan(z['z']/vec4(z['x'], z['y']).length());
	}
};

class LineStrip {
		GLuint vao, vbo;        // vertex array object, vertex buffer object
		float  vertexData[10000]; // interleaved data of coordinates and colors
		int    nVertices;       // number of vertices
	public:
		LineStrip() {
			nVertices = 0;
		}
		void Create() {
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);
	
			glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			// Enable the vertex attribute arrays
			glEnableVertexAttribArray(0);  // attribute array 0
			glEnableVertexAttribArray(1);  // attribute array 1
			// Map attribute array 0 to the vertex data of the interleaved vbo
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
			// Map attribute array 1 to the color data of the interleaved vbo
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
		}
	
		void clear() {
			nVertices = 0;
		}

		void AddPoint(float cX, float cY) {
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			if (nVertices >= 2000) return;
	
			vec4 wVertex = vec4(cX, cY, 0, 1);
			// fill interleaved data
			vertexData[5 * nVertices]     = wVertex.v[0];
			vertexData[5 * nVertices + 1] = wVertex.v[1];
			vertexData[5 * nVertices + 2] = 1; // red
			vertexData[5 * nVertices + 3] = 1; // green
			vertexData[5 * nVertices + 4] = 0; // blue
			nVertices++;
			// copy data to the GPU
			glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
		}
	
		void Draw() {
			if (nVertices > 0) {
				/*mat4 VPTransform = mat4();
	
				int location = glGetUniformLocation(shaderProgram, "MVP");
				if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
				else printf("uniform MVP cannot be set\n");*/
	
				glBindVertexArray(vao);
				glDrawArrays(GL_LINE_STRIP, 0, nVertices);
			}
		}
	};

class LagrangeSpline {
	vector<vec4> controlPoints;
	vector<float> controlTimes;
	vector<vec4> interpolatedPoints;
	LineStrip controlLine;
	LineStrip interpolatedLine;

	float weight(size_t k, float t) {
		float result = 1;
		for (size_t i = 0; i < controlTimes.size(); i++) {
			if (i != k) {
				result *= (t - controlTimes[i]) / (controlTimes[k] - controlTimes[i]);
			}
		}
		return result;
	}

	float derivatedWeight(size_t k, float t) {
		float result = 0;
		for (size_t i = 0; i < controlTimes.size(); i++) {
			if (i != k) {
				result += 1 / (t - controlTimes[i]);
			}
		}
		return result*weight(k,t);
	}


	void createInterpolatedPoints() {
		// TODO

		interpolatedLine = LineStrip();
		interpolatedLine.Create();
		interpolatedLine.clear();
		for (size_t i = 0; i < controlPoints.size() - 1; i++) {
			for (size_t j = 0; j < 20; j++) {
				interpolatedPoints.push_back(getPositionAtTime(
					controlTimes[i] + (controlTimes[i+1]- controlTimes[i])/20*j
				));
				interpolatedLine.AddPoint(interpolatedPoints.back()['x'], interpolatedPoints.back()['y']);
			}
		}
		interpolatedLine.AddPoint(controlPoints.back()['x'], controlPoints.back()['y']);
	}

	void createControlLine() {
		// TODO
		controlLine = LineStrip();
		controlLine.Create();
		for (size_t i = 0; i < controlPoints.size(); i++) {
			controlLine.AddPoint(controlPoints[i]['x'], controlPoints[i]['y']);
		}
	}
	vec4 getPositionAtTime(float t) {
		vec4 position;
		for (size_t i = 0; i < controlPoints.size(); i++) {
			position += controlPoints[i] * weight(i, t);
		}
		return position;
	}
	vec4 getDirectionAtTime(float t) {
		vec4 position;
		for (size_t i = 0; i < controlPoints.size(); i++) {
			position += controlPoints[i] * derivatedWeight(i, t);
		}
		return position;
	}
public:
	void addControlPoint(vec4 newPoint, float time) {
		controlPoints.push_back(newPoint);
		controlTimes.push_back(time);
		createControlLine();
		interpolatedPoints.clear();
		createInterpolatedPoints();
	}

	void draw() {
		//controlLine.Draw();
		interpolatedLine.Draw();
	}

	vec4 getPositionAtRelativeTime(float t) {
		return getPositionAtTime(t + controlTimes[0]);
	}

	vec4 getDirectionAtRelativeTime(float t) {
		return getDirectionAtTime(t + controlTimes[0]);
	}

	float getTimeLength() {
		return controlTimes.back() - controlTimes[0];
	}

	size_t getNumberOfControls() { return controlTimes.size(); }
};


BezierSurface BS;
LagrangeSpline LS;

class Bike {
	vector<Triangle3D> triangles;
public:
	void animate(float t) {
		vec4 p1;
		vec4 p2(1, -1);
		vec4 p3(0, 1);
		vec4 p4(-1, -1);

		vec4 pos = LS.getPositionAtRelativeTime(t);
		vec4 dir = LS.getDirectionAtRelativeTime(t).normalize();
		dir = vec4(dir['y'], dir['x']);
		//cout << "pos: " << pos['x'] << " : " << pos['y'] << endl;
		//cout << BS.getRollPosition((pos['x'] + 500) / 1000.0, (pos['x'] + 500) / 1000.0, dir)<<endl;
		mat4 T(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			pos['x'], pos['y'], 0, 1
		);

		mat4 S(
			20, 0, 0, 0,
			0, 20, 0, 0,
			0, 0, 20, 0,
			0,0, 0, 1
		);

		mat4 R(
			dir['x'], -dir['y'], 0, 0,
			dir['y'], dir['x'], 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		);
		mat4 M = S*R*T;
		if (triangles.size() < 2) { triangles.push_back(Triangle3D(p1*M, p2*M, p3*M)); triangles.push_back(Triangle3D(p1*M, p3*M, p4*M)); }
		triangles[0] = Triangle3D(p1*M, p2*M, p3*M);
		triangles[1] = Triangle3D(p1*M, p3*M, p4*M);
	}

	void draw() {
		for (size_t i = 0; i < triangles.size(); i++) {
			triangles[i].Draw();
		}
	}
};


vector<Point*> points;

Bike bike;

//LineStrip lineStrip;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);

	camera = new Camera3D();

	BS.createControlSurface();
	BS.createInterpolatedSurface();
	BS.createGrads();
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	BS.draw();
	for (size_t i = 0; i < points.size(); i++) {
		points[i]->draw();
	}
	LS.draw();
	bike.draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;

		long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
		float sec = time / 1000.0f;				// convert msec to sec

		points.push_back(new Point(vec4(
			cX * 500,
			cY * 500,
			0
			)));
		LS.addControlPoint(vec4(
			cX * 500,
			cY * 500,
			0
		), sec);
		//lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	
	if(LS.getNumberOfControls()>1 && time%10==0)
		bike.animate((sec/10 - floor(sec/10 / LS.getTimeLength())*LS.getTimeLength()));

	camera->Animate(sec);


	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

