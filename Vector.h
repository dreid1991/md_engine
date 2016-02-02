#ifndef VECTOR_H
#define VECTOR_H
#include "Python.h"
#include <math.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include "globalDefs.h"
#include "cutils_math.h"
using namespace std;
#define EPSILON .000001

#include "boost_for_export.h"
void export_Vector();
void export_VectorInt();

template <typename T, typename K>
class VectorGeneric;

typedef VectorGeneric<num, int> Vector;
typedef VectorGeneric<int, num> VectorInt;

//theme for these operations is that if we're operating on unlike types, switch to num representation.  would like better way to do this
template <typename T, typename K>
class VectorGeneric {
	T vals[3];
	public:

		VectorGeneric<T, K> () {
			vals[0] = vals[1] = vals[2] = 0;
		}
		VectorGeneric<T, K> (num x, num y, num z) {  // if using literals, have to cast each as int or num so compiler can distinguish between these two constructors
			vals[0] = x;
			vals[1] = y;
			vals[2] = z;
		}
		VectorGeneric<T, K> (int vals_[3]) {
			for (int i=0; i<3; i++) {
				vals[i] = (T) vals_[i];
			}
		}
		VectorGeneric<T, K> (num vals_[3]) {
			for (int i=0; i<3; i++) {
				vals[i] = (T) vals_[i];
			}
		}
        VectorGeneric<T, K> (float3 other) {
            vals[0] = other.x;
            vals[1] = other.y;
            vals[2] = other.z;
        }
        VectorGeneric<T, K> (float4 other) {
            vals[0] = other.x;
            vals[1] = other.y;
            vals[2] = other.z;
        }
        float4 asFloat4() {
            return make_float4(vals[0], vals[1], vals[2], 0);
        }
        int4 asInt4() {
            return make_int4(vals[0], vals[1], vals[2], 0);
        }
        float3 asFloat3() {
            return make_float3(vals[0], vals[1], vals[2]);
        }
        int3 asInt3() {
            return make_int3(vals[0], vals[1], vals[2]);
        }
		void zero() {
			vals[0] = vals[1] = vals[2] = 0;
		}
        /*
		void skew(num cao, num sao, num cbo, num sbo, num caf, num saf, num cbf, num sbf) { // c->cos, s->sin, a->alphe, b->beta, o->orig, f->final
			num x = vals[0]; //See mathematica notebook for this math.  Basically untransforming to no rotation, then transforming to new coords
			num y = vals[1];
			num denom = cao*cbo - sao*sbo;
			vals[0] = ((cao*y-x*sao)*sbf + caf*(cbo*x-y*sbo)) / denom;
			vals[1] = (cbo*x*saf + cbf * (cao*y - x*sao) - y*saf*sbo) / denom;

		}
		void skew(num ao, num bo, num af, num bf) {
			skew(cos(ao), sin(ao), cos(bo), sin(bo), cos(af), sin(af), cos(bf), sin(bf));
		}
        void skewPy(num ao, num bo, num af, num bf) {
            skew(ao, bo, af, bf);
        }
		void skewFromZero(num ca, num sa, num cb, num sb) {
			num xo = vals[0];
			num yo = vals[1];
			vals[0] = xo*ca + yo*sb;
			vals[1] = xo*sa + yo*cb;
		}
		void unskew(num ca, num sa, num cb, num sb) {
			num denom = ca*cb - sa*sb;
			num xo = vals[0];
			num yo = vals[1];
			vals[0] = (cb*xo - sb*yo) / denom;
			vals[1] = (ca*yo - sa*xo) / denom;
		}
        */
		VectorGeneric<T, K> VTo(const VectorGeneric<T, K> &v) {
			return VectorGeneric<T, K>(v[0] - vals[0], v[1] - vals[1], v[2] - vals[2]);
		}
		VectorGeneric<num, int> VTo(const VectorGeneric<K, T> &v) {
			return VectorGeneric<num, int>(v[0] - vals[0], v[1] - vals[1], v[2] - vals[2]);
		}
		T sum() {
			return vals[0] + vals[1] + vals[2];
		}
		T prod() {
			return vals[0] * vals[1] * vals[2];
		}
		T &operator[]( int n ){
			return vals[n]; 
		}
		T operator[]( const int n )const{
			return vals[n];
		}

		VectorGeneric<T, K> abs() {
			return VectorGeneric<T, K>((T) fabs(vals[0]), (T) fabs(vals[1]), (T) fabs(vals[2]));
		}
		VectorGeneric<T, K> operator-()const{
			return VectorGeneric<T, K>(-vals[0], -vals[1], -vals[2]); 
		}
		VectorGeneric<num, int> operator*( num scale )const{
			return VectorGeneric<num, int>( vals[0]*scale,vals[1]*scale,vals[2]*scale );
		}

		VectorGeneric<T, K> operator*( const VectorGeneric<T, K> &q )const{
			return VectorGeneric<T, K>( vals[0]*q[0],vals[1]*q[1],vals[2]*q[2]);
		}
		VectorGeneric<num, int> operator*( const VectorGeneric<K, T> &q )const{
			return VectorGeneric<num, int>( vals[0]*q[0],vals[1]*q[1],vals[2]*q[2]);
		}

		VectorGeneric<T, K> operator/( int scale )const{
			return VectorGeneric<T, K>( vals[0]/scale,vals[1]/scale,vals[2]/scale );
		}
		VectorGeneric<num, int> operator/( num scale )const{
			return VectorGeneric<num, int>( vals[0]/scale,vals[1]/scale,vals[2]/scale );
		}
		VectorGeneric<num, int> rotate2d( num rotation) const {
			num c = cos(rotation);
			num s = sin(rotation);
			return VectorGeneric<num, int> (c*vals[0] - s*vals[1], s*vals[0] + c*vals[1], vals[2]);
		}
		VectorGeneric<T, K> operator/( const VectorGeneric<T, K> &q )const{
			return VectorGeneric<T, K>( vals[0]/q[0],vals[1]/q[1],vals[2]/q[2] );
		}
		VectorGeneric<num, int> operator/( const VectorGeneric<K, T> &q )const{
			return VectorGeneric<num, int>( vals[0]/q[0],vals[1]/q[1],vals[2]/q[2] );
		}

		VectorGeneric<T, K> operator+( const VectorGeneric<T, K> &q )const{
			return VectorGeneric<T, K>( vals[0]+q[0],vals[1]+q[1],vals[2]+q[2] );
		}
		VectorGeneric<num, int> operator+( const VectorGeneric<K, T> &q )const{
			return VectorGeneric<num, int>( vals[0]+q[0],vals[1]+q[1],vals[2]+q[2] );
		}

		VectorGeneric<T, K> operator-( const VectorGeneric<T, K> &q )const{
			return VectorGeneric<T, K>( vals[0]-q[0],vals[1]-q[1],vals[2]-q[2] );
		}
		VectorGeneric<num, int> operator-( const VectorGeneric<K, T> &q )const{
			return VectorGeneric<num, int>( vals[0]-q[0],vals[1]-q[1],vals[2]-q[2] );
		}

		VectorGeneric<T, K> &operator*=( int scale ){
			vals[0]*=scale;vals[1]*=scale;vals[2]*=scale;return *this; // *=, /=, etc won't promote types like binary operations
		}
		VectorGeneric<T, K> &operator*=( num scale ){
			vals[0]*=scale;vals[1]*=scale;vals[2]*=scale;return *this;
		}

		VectorGeneric<T, K> &operator*=( const VectorGeneric<T, K> &q ){
			vals[0]*=q[0];vals[1]*=q[1];vals[2]*=q[2];return *this;
		}
		VectorGeneric<T, K> &operator*=( const VectorGeneric<K, T> &q ){
			vals[0]*=q[0];vals[1]*=q[1];vals[2]*=q[2];return *this;
		}

		VectorGeneric<T, K> &operator/=( int scale ){
			vals[0]/=scale;vals[1]/=scale;vals[2]/=scale;return *this;
		}
		VectorGeneric<T, K> &operator/=( num scale ){
			vals[0]/=scale;vals[1]/=scale;vals[2]/=scale;return *this;
		}

		VectorGeneric<T, K> &operator/=( const VectorGeneric<T, K> &q ){
			vals[0]/=q[0];vals[1]/=q[1];vals[2]/=q[2];return *this;
		}
		VectorGeneric<T, K> &operator/=( const VectorGeneric<K, T> &q ){
			vals[0]/=q[0];vals[1]/=q[1];vals[2]/=q[2];return *this;
		}

		VectorGeneric<T, K> &operator+=( const VectorGeneric<T, K> &q ){
			vals[0]+=q[0];vals[1]+=q[1];vals[2]+=q[2];return *this;
		}
		VectorGeneric<T, K> &operator+=( const VectorGeneric<K, T> &q ){
			vals[0]+=q[0];vals[1]+=q[1];vals[2]+=q[2];return *this;
		}

		VectorGeneric<T, K> &operator-=( const VectorGeneric<T, K> &q ){
			vals[0]-=q[0];vals[1]-=q[1];vals[2]-=q[2];return *this;
		}
		VectorGeneric<T, K> &operator-=( const VectorGeneric<K, T> &q ){
			vals[0]-=q[0];vals[1]-=q[1];vals[2]-=q[2];return *this;
		}
		bool operator<( const VectorGeneric<T, K> &q )const{
			if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]<q[0] ? true : false;
			if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]<q[1] ? true : false;
			return fabs(vals[2]-q[2])>EPSILON && vals[2]<q[2];
		}
		bool operator<( const VectorGeneric<K, T> &q )const{
			if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]<q[0] ? true : false;
			if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]<q[1] ? true : false;
			return fabs(vals[2]-q[2])>EPSILON && vals[2]<q[2];
		}

		bool operator>( const VectorGeneric<T, K> &q )const{
			if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]>q[0] ? true : false;
			if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]>q[1] ? true : false;
			return fabs(vals[2]-q[2])>EPSILON && vals[2]>q[2];
		}
		bool operator>( const VectorGeneric<K, T> &q )const{
			if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]>q[0] ? true : false;
			if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]>q[1] ? true : false;
			return fabs(vals[2]-q[2])>EPSILON && vals[2]<q[2];
		}
		bool operator==( const VectorGeneric<T, K> &q )const{
			return fabs(vals[0]-q[0])<=EPSILON && fabs(vals[1]-q[1])<=EPSILON && fabs(vals[2]-q[2])<=EPSILON;
		}
		bool operator==( const VectorGeneric<K, T> &q )const{
			return fabs(vals[0]-q[0])<=EPSILON && fabs(vals[1]-q[1])<=EPSILON && fabs(vals[2]-q[2])<=EPSILON;
		}

		bool operator!=( const VectorGeneric<T, K> &q )const{
			return fabs(vals[0]-q[0])>EPSILON || fabs(vals[1]-q[1])>EPSILON || fabs(vals[2]-q[2])>EPSILON;
		}
		bool operator!=( const VectorGeneric<K, T> &q )const{
			return fabs(vals[0]-q[0])>EPSILON || fabs(vals[1]-q[1])>EPSILON || fabs(vals[2]-q[2])>EPSILON;
		}

		num dot( const VectorGeneric<T, K> &q )const{
			return vals[0]*q[0]+vals[1]*q[1]+vals[2]*q[2];
		}
		num dot( const VectorGeneric<K, T> &q )const{
			return vals[0]*q[0]+vals[1]*q[1]+vals[2]*q[2];
		}
		VectorGeneric<T, K> cross( const VectorGeneric<T, K> &q )const{
			return VectorGeneric<T, K>( vals[1]*q[2]-vals[2]*q[1],vals[2]*q[0]-vals[0]*q[2],vals[0]*q[1]-vals[1]*q[0] );
		}
		VectorGeneric<num, int> cross( const VectorGeneric<K, T> &q )const{
			return VectorGeneric<num, int>( vals[1]*q[2]-vals[2]*q[1],vals[2]*q[0]-vals[0]*q[2],vals[0]*q[1]-vals[1]*q[0] );
		}
		num len()const{
			return sqrt((num) vals[0]*vals[0]+vals[1]*vals[1]+vals[2]*vals[2]);
		}
		num lenSqr()const{
			return vals[0]*vals[0]+vals[1]*vals[1]+vals[2]*vals[2];
		}
		num dist( const VectorGeneric<T, K> &q )const{
			num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return sqrt(dx*dx+dy*dy+dz*dz);
		}
		num dist( const VectorGeneric<K, T> &q )const{
			num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return sqrt(dx*dx+dy*dy+dz*dz);
		}
		num distPython( const VectorGeneric<T, K> &q)const{
			num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return sqrt(dx*dx+dy*dy+dz*dz);
		}
		num distSqr( const VectorGeneric<T, K> &q) {
			num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return dx*dx+dy*dy+dz*dz;
		}
		num distSqr( const VectorGeneric<K, T> &q) {
			num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return dx*dx+dy*dy+dz*dz;
		}

		VectorGeneric<num, int> normalized()const{
			num l=len();return VectorGeneric<num, int>( vals[0]/l,vals[1]/l,vals[2]/l );
		}
		void normalize(){
			num l=len();vals[0]/=l;vals[1]/=l;vals[2]/=l; //will not necessarily normalize int vectors
		}
		VectorGeneric<T, K> perp2d() {
			return VectorGeneric(vals[1], -vals[0], vals[2]);
		}
		void clear(){
			vals[0] = vals[1] = vals[2] = 0;
		}

		string asStr() const {
			char buffer[100];
			num x = vals[0], y = vals[1], z = vals[2];
			int n = sprintf(buffer, "x: %f, y: %f, z: %f", x, y, z);
			assert(n >= 0);	
			return string(buffer);
		}
		VectorGeneric<T, K> &operator=(const VectorGeneric<K, T> &other) {
			for (int i=0; i<3; i++) {
				vals[i] = other[i];
			}
			return *this;
		}
        VectorGeneric<T, K> &operator=(const float3 &other) {
            vals[0] = other.x;
            vals[1] = other.y;
            vals[2] = other.z;
            return *this;
        }
        VectorGeneric<T, K> &operator=(const float4 &other) {
            vals[0] = other.x;
            vals[1] = other.y;
            vals[2] = other.z;
            return *this;
        }
		VectorGeneric<T, K> (const VectorGeneric<K, T> &other) {
			for (int i=0; i<3; i++) {
				vals[i] = other[i];
			}
		}
		T get(int i) {
			return vals[i];
		}
		void set(int i, T val) {
			vals[i] = val;
		}
		VectorGeneric<T, K> loopedVTo(const VectorGeneric<T, K> &other, const VectorGeneric<T, K> &trace) {
			VectorGeneric<T, K> dist = other - *this;
			VectorGeneric<T, K> halfTrace = trace/ (T) 2.0;
			for (int i=0; i<3; i++) {
				if (dist[i] > halfTrace[i]) {
					dist[i] -= trace[i];
				} else if (dist[i] < -halfTrace[i]) {
					dist[i] += trace[i];
				}
			}
			return dist;

		}
};


std::ostream &operator<<(std::ostream &os, const Vector &v);
std::ostream &operator<<(std::ostream &os, const float4 &v);

#endif
