//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiv�ve
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

	//uniform mat4 MVP;			// Model-View-Projection matrix in row-major format
	uniform mat4 ModelWorldScale;
	uniform mat4 ModelWorldRotation;
	uniform mat4 ModelWorldTranslation;
	uniform mat4 WorldView;

	layout(location = 0) in vec3 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vec3(1.0,1.0,1.0);//vec3(vertexPosition.z/2.0,vertexPosition.z/2.0,vertexPosition.z/2.0);//vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, vertexPosition.z, 1)
						*ModelWorldScale
						*ModelWorldRotation
						*ModelWorldTranslation
						*WorldView;
						
		 //* MVP; 		// transform to clipping space
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

	float operator[](char c) {
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
		scale = vec4(20,20,20);
		rotation = vec4(-0.3, t, 0);

		// commit uniform variables
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
			0 - position['x'], 0 - position['y'] + 5, 0 - position['z'], 1
		);
		mat4 WorldViewScale(
			1/scale['x'], 0, 0, 0,
			0, 1 / scale['y'], 0, 0,
			0, 0, 1 / scale['z'], 0,
			0, 0, 0, 1
		);
		mat4 WorldView = WorldViewTranslation*WorldViewRotationY*WorldViewRotationX*WorldViewScale;

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
//
//class LineStrip {
//	GLuint vao, vbo;        // vertex array object, vertex buffer object
//	float  vertexData[100]; // interleaved data of coordinates and colors
//	int    nVertices;       // number of vertices
//public:
//	LineStrip() {
//		nVertices = 0;
//	}
//	void Create() {
//		glGenVertexArrays(1, &vao);
//		glBindVertexArray(vao);
//
//		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
//		glBindBuffer(GL_ARRAY_BUFFER, vbo);
//		// Enable the vertex attribute arrays
//		glEnableVertexAttribArray(0);  // attribute array 0
//		glEnableVertexAttribArray(1);  // attribute array 1
//		// Map attribute array 0 to the vertex data of the interleaved vbo
//		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
//		// Map attribute array 1 to the color data of the interleaved vbo
//		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
//	}
//
//	void AddPoint(float cX, float cY) {
//		glBindBuffer(GL_ARRAY_BUFFER, vbo);
//		if (nVertices >= 20) return;
//
//		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
//		// fill interleaved data
//		vertexData[5 * nVertices]     = wVertex.v[0];
//		vertexData[5 * nVertices + 1] = wVertex.v[1];
//		vertexData[5 * nVertices + 2] = 1; // red
//		vertexData[5 * nVertices + 3] = 1; // green
//		vertexData[5 * nVertices + 4] = 0; // blue
//		nVertices++;
//		// copy data to the GPU
//		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
//	}
//
//	void Draw() {
//		if (nVertices > 0) {
//			mat4 VPTransform = camera.V() * camera.P();
//
//			int location = glGetUniformLocation(shaderProgram, "MVP");
//			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
//			else printf("uniform MVP cannot be set\n");
//
//			glBindVertexArray(vao);
//			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
//		}
//	}
//};


std::vector<Triangle3D*> triangles = std::vector<Triangle3D*>();
//LineStrip lineStrip;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	//lineStrip.Create();
	//triangle.Create(vec4(10, 10, 0), vec4(10, -10, 0.75), vec4(-10, -10, 1.5));
	//triangle2.Create(vec4(-10,10,0), vec4(0, 10, 0.75), vec4(-10, 0, 1.5));

	

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

	triangles.push_back(new Triangle3D(vec4(5, 5, -5), vec4(5, -5, -5), vec4(-5, -5, -5)));
	triangles.push_back(new Triangle3D(vec4(5, 5, -5), vec4(-5, 5, -5), vec4(-5, -5, -5)));
	triangles.push_back(new Triangle3D(vec4(-5, -5, -5), vec4(-5, -5, 5), vec4(5, -5, -5)));
	triangles.push_back(new Triangle3D(vec4(5, -5, 5), vec4(-5, -5, 5), vec4(5, -5, -5)));

}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	for (unsigned int i = 0; i < triangles.size(); i++) {
		triangles[i]->Draw();
	}
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
	camera->Animate(sec);					// animate the camera
	//triangle.Animate(sec);					// animate the triangle object
	//triangle2.Animate(sec);					// animate the triangle object
	for (unsigned int i = 0; i < triangles.size(); i++) {
		triangles[i]->Animate(sec);
	}
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
