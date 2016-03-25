#ifndef VECTOR_H
#define VECTOR_H

#include "Python.h"
#include <math.h>
#include <string>
#include <sstream>
#include <iostream>
#include "globalDefs.h"
#include "cutils_math.h"

using namespace std;
#define EPSILON .000001f

#include "boost_for_export.h"
void export_Vector();
void export_VectorInt();

template <typename T, typename K>
class VectorGeneric;

typedef VectorGeneric<num, int> Vector;
typedef VectorGeneric<int, num> VectorInt;

/*! \class VectorGeneric
 * \brief A three-element vector
 *
 * \tparam T Type of data stored in the vector.
 * \tparam K I have no idea.
 *
 * This class defines a simple three-element vector and the corresponding
 * vector operations.
 *
 * \todo Remove usage of typename K. Is it even used? And if yes, what does
 *       it represent? This is very confusing.
 */
//theme for these operations is that if we're operating on unlike types, switch
//to num representation.  would like better way to do this
template <typename T, typename K>
class VectorGeneric {
	T vals[3]; //!< Array storing the values
public:

    /*! \brief Default constructor */
    VectorGeneric<T, K> () {
        vals[0] = vals[1] = vals[2] = 0;
    }

    // if using literals, have to cast each as int or num so compiler can
    // distinguish between these two constructors
    /*! \brief Constructor
     *
     * \param x First element as double
     * \param y Second element as double
     * \param z Third element as double
     *
     * \todo It would be much more sensible to have
     *       VectorGeneric<T, K> (T x, T y, T z)
     */
    VectorGeneric<T, K> (num x, num y, num z) {
        vals[0] = x;
        vals[1] = y;
        vals[2] = z;
    }

    /*! \brief Constructor from int pointer
     *
     * \param vals_ Pointer to three element int array
     *
     * \todo The [3] specifier does not do anything. It is still just a
     *       pointer being passed to the function.
     */
    VectorGeneric<T, K> (int vals_[3]) {
        for (int i=0; i<3; i++) {
            vals[i] = (T) vals_[i];
        }
    }

    /*! \brief Constructor from double pointer
     *
     * \param vals_ Pointer to three element double array
     *
     * \todo The [3] specifier does nothing. It is still just a pointer being
     *       passed to the constructor.
     */
    VectorGeneric<T, K> (num vals_[3]) {
        for (int i=0; i<3; i++) {
            vals[i] = (T) vals_[i];
        }
    }

    /*! \brief Constructor from float3
     *
     * \param other Float3 to use as values for the vector
     */
    VectorGeneric<T, K> (float3 other) {
        vals[0] = other.x;
        vals[1] = other.y;
        vals[2] = other.z;
    }

    /*! \brief Constructor from float4
     *
     * \param other Float4 to use as values for the vector
     *
     * The forth value in the float4 will be discarded.
     */
    VectorGeneric<T, K> (float4 other) {
        vals[0] = other.x;
        vals[1] = other.y;
        vals[2] = other.z;
    }

    /*! \brief Convert vector to float4
     *
     * The first three entries correspond to the vector elements, the forth
     * entry will be set to zero.
     */
    float4 asFloat4() {
        return make_float4(vals[0], vals[1], vals[2], 0);
    }

    /*! \brief Convert vector to int4
     *
     * The first three entries correspond to the vector elements, the forth
     * entry will be set to zero.
     */
    int4 asInt4() {
        return make_int4(vals[0], vals[1], vals[2], 0);
    }

    /*! \brief Convert vector to float3 */
    float3 asFloat3() {
        return make_float3(vals[0], vals[1], vals[2]);
    }

    /*! \brief Convert vector to int3 */
    int3 asInt3() {
        return make_int3(vals[0], vals[1], vals[2]);
    }

    /*! \brief Set all vector elements to zero */
    void zero() {
        vals[0] = vals[1] = vals[2] = 0;
    }

    //void skew(num cao, num sao, num cbo, num sbo, num caf, num saf, num cbf, num sbf) { // c->cos, s->sin, a->alphe, b->beta, o->orig, f->final
    //    num x = vals[0]; //See mathematica notebook for this math.  Basically untransforming to no rotation, then transforming to new coords
    //    num y = vals[1];
    //    num denom = cao*cbo - sao*sbo;
    //    vals[0] = ((cao*y-x*sao)*sbf + caf*(cbo*x-y*sbo)) / denom;
    //    vals[1] = (cbo*x*saf + cbf * (cao*y - x*sao) - y*saf*sbo) / denom;
    //}
    //void skew(num ao, num bo, num af, num bf) {
    //    skew(cos(ao), sin(ao), cos(bo), sin(bo), cos(af), sin(af), cos(bf), sin(bf));
    //}
    //void skewPy(num ao, num bo, num af, num bf) {
    //    skew(ao, bo, af, bf);
    //}
    //void skewFromZero(num ca, num sa, num cb, num sb) {
    //    num xo = vals[0];
    //    num yo = vals[1];
    //    vals[0] = xo*ca + yo*sb;
    //    vals[1] = xo*sa + yo*cb;
    //}
    //void unskew(num ca, num sa, num cb, num sb) {
    //    num denom = ca*cb - sa*sb;
    //    num xo = vals[0];
    //    num yo = vals[1];
    //    vals[0] = (cb*xo - sb*yo) / denom;
    //    vals[1] = (ca*yo - sa*xo) / denom;
    //}

    /*! \brief Vector connecting two points
     *
     * \param v Other point
     * \returns A new vector connecting the points defined by this vector and
     *          vector v
     *
     * This function gives the vector from the point specified by vector v to
     * the point this vector is pointing to.
     *
     * \todo This is essentially (v - *this). Do we really need this function?
     */
    VectorGeneric<T, K> VTo(const VectorGeneric<T, K> &v) {
        return VectorGeneric<T, K>(v[0] - vals[0], v[1] - vals[1], v[2] - vals[2]);
    }

    /*! \brief Vector connecting two points
     *
     * \param v Other point
     * \returns A vector storing doubles connecting the points defined by this
     *          vector and vector v
     *
     * This function returns the vector from to the point vector v is pointing
     * to the point this vector is pointing to. The returned vector stores
     * doubles, the conversion takes place after the operation, however.
     *
     * \todo This is essentially (v -*this) converted to <num, int>. Do we
     *       really need this function?
     * \todo Why is the explicit conversion necessary? This is exactly what is
     *       done implicitly.
     */
    VectorGeneric<num, int> VTo(const VectorGeneric<K, T> &v) {
        return VectorGeneric<num, int>(v[0] - vals[0], v[1] - vals[1], v[2] - vals[2]);
    }

    /*! \brief Sum of all entries */
    T sum() {
        return vals[0] + vals[1] + vals[2];
    }

    /*! \brief Product of all entries */
    T prod() {
        return vals[0] * vals[1] * vals[2];
    }

    /*! \brief Operator accessing vector elements */
    T &operator[]( int n ){
        return vals[n];
    }

    /*! \brief Const operator accessing vector elements
     *
     * \todo The const int is not necessary. Returning const T & would be more
     *       sensible (It isn't much more efficient, but more the typical
     *       structure of access operators).
     */
    T operator[]( const int n )const{
        return vals[n];
    }

    /*! \brief Convert all elements to their absolute value
     *
     * \returns A new vector with the transformed elements.
     */
    VectorGeneric<T, K> abs() {
        return VectorGeneric<T, K>((T) fabs(vals[0]), (T) fabs(vals[1]), (T) fabs(vals[2]));
    }

    /*! \brief Unary minus operator */
    VectorGeneric<T, K> operator-()const{
        return VectorGeneric<T, K>(-vals[0], -vals[1], -vals[2]);
    }

    /*! \brief Multiplication with double operator */
    VectorGeneric<num, int> operator*( num scale )const{
        return VectorGeneric<num, int>( vals[0]*scale,vals[1]*scale,vals[2]*scale );
    }

    /*! \brief Multiplication with other vector operator */
    VectorGeneric<T, K> operator*( const VectorGeneric<T, K> &q )const{
        return VectorGeneric<T, K>( vals[0]*q[0],vals[1]*q[1],vals[2]*q[2]);
    }

    /*! \brief Multiplication with other vector where K and T are exchanged */
    VectorGeneric<num, int> operator*( const VectorGeneric<K, T> &q )const{
        return VectorGeneric<num, int>( vals[0]*q[0],vals[1]*q[1],vals[2]*q[2]);
    }

    /*! \brief Division with int operator */
    VectorGeneric<T, K> operator/( int scale )const{
        return VectorGeneric<T, K>( vals[0]/scale,vals[1]/scale,vals[2]/scale );
    }

    /*! \brief Division with double operator */
    VectorGeneric<num, int> operator/( num scale )const{
        return VectorGeneric<num, int>( vals[0]/scale,vals[1]/scale,vals[2]/scale );
    }

    /*! \brief rotation in x-y plane
     *
     * \param rotation Rotation angle
     *
     * The z-component of the vector remains unchanged.
     */
    VectorGeneric<num, int> rotate2d( num rotation) const {
        num c = cos(rotation);
        num s = sin(rotation);
        return VectorGeneric<num, int> (c*vals[0] - s*vals[1], s*vals[0] + c*vals[1], vals[2]);
    }

    /*! \brief Element-wise division */
    VectorGeneric<T, K> operator/( const VectorGeneric<T, K> &q )const{
        return VectorGeneric<T, K>( vals[0]/q[0],vals[1]/q[1],vals[2]/q[2] );
    }

    /*! \brief Element-wise division */
    VectorGeneric<num, int> operator/( const VectorGeneric<K, T> &q )const{
        return VectorGeneric<num, int>( vals[0]/q[0],vals[1]/q[1],vals[2]/q[2] );
    }

    /*! \brief Addition of two vectors */
    VectorGeneric<T, K> operator+( const VectorGeneric<T, K> &q )const{
        return VectorGeneric<T, K>( vals[0]+q[0],vals[1]+q[1],vals[2]+q[2] );
    }

    /*! \brief Addition of two vectors */
    VectorGeneric<num, int> operator+( const VectorGeneric<K, T> &q )const{
        return VectorGeneric<num, int>( vals[0]+q[0],vals[1]+q[1],vals[2]+q[2] );
    }

    /*! \brief Subtraction of two vectors */
    VectorGeneric<T, K> operator-( const VectorGeneric<T, K> &q )const{
        return VectorGeneric<T, K>( vals[0]-q[0],vals[1]-q[1],vals[2]-q[2] );
    }

    /*! \brief Subtraction of two vectors */
    VectorGeneric<num, int> operator-( const VectorGeneric<K, T> &q )const{
        return VectorGeneric<num, int>( vals[0]-q[0],vals[1]-q[1],vals[2]-q[2] );
    }

    /*! \brief Multiplication-assignment operator with int */
    VectorGeneric<T, K> &operator*=( int scale ){
        vals[0]*=scale;vals[1]*=scale;vals[2]*=scale;return *this; // *=, /=, etc won't promote types like binary operations
    }

    /*! \brief Multiplication-assignment operator with double */
    VectorGeneric<T, K> &operator*=( num scale ){
        vals[0]*=scale;vals[1]*=scale;vals[2]*=scale;return *this;
    }

    /*! \brief Multiplication-assignment operator with other vector
     *
     * Performs element-wise multiplication.
     */
    VectorGeneric<T, K> &operator*=( const VectorGeneric<T, K> &q ){
        vals[0]*=q[0];vals[1]*=q[1];vals[2]*=q[2];return *this;
    }

    /*! \brief Multiplication-assignment operator
     *
     * Performs element-wise multiplication.
     */
    VectorGeneric<T, K> &operator*=( const VectorGeneric<K, T> &q ){
        vals[0]*=q[0];vals[1]*=q[1];vals[2]*=q[2];return *this;
    }

    /*! \brief Division-assignment operator with int */
    VectorGeneric<T, K> &operator/=( int scale ){
        vals[0]/=scale;vals[1]/=scale;vals[2]/=scale;return *this;
    }

    /*! \brief Division-assignment operator with double */
    VectorGeneric<T, K> &operator/=( num scale ){
        vals[0]/=scale;vals[1]/=scale;vals[2]/=scale;return *this;
    }

    /*! \brief Division-assignment operator with other vector
     *
     * Performs element-wise division.
     */
    VectorGeneric<T, K> &operator/=( const VectorGeneric<T, K> &q ){
        vals[0]/=q[0];vals[1]/=q[1];vals[2]/=q[2];return *this;
    }

    /*! \brief Division-assignment operator with other vector
     *
     * Performs element-wise division.
     */
    VectorGeneric<T, K> &operator/=( const VectorGeneric<K, T> &q ){
        vals[0]/=q[0];vals[1]/=q[1];vals[2]/=q[2];return *this;
    }

    /*! \brief Addition-assignment operator */
    VectorGeneric<T, K> &operator+=( const VectorGeneric<T, K> &q ){
        vals[0]+=q[0];vals[1]+=q[1];vals[2]+=q[2];return *this;
    }

    /*! \brief Addition-assignment operator */
    VectorGeneric<T, K> &operator+=( const VectorGeneric<K, T> &q ){
        vals[0]+=q[0];vals[1]+=q[1];vals[2]+=q[2];return *this;
    }

    /*! \brief Subtraction-assigment operator */
    VectorGeneric<T, K> &operator-=( const VectorGeneric<T, K> &q ){
        vals[0]-=q[0];vals[1]-=q[1];vals[2]-=q[2];return *this;
    }

    /*! \brief Subtraction-assigment operator */
    VectorGeneric<T, K> &operator-=( const VectorGeneric<K, T> &q ){
        vals[0]-=q[0];vals[1]-=q[1];vals[2]-=q[2];return *this;
    }

    /*! \brief Smaller than comparison operator */
    bool operator<( const VectorGeneric<T, K> &q )const{
        if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]<q[0] ? true : false;
        if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]<q[1] ? true : false;
        return fabs(vals[2]-q[2])>EPSILON && vals[2]<q[2];
    }

    /*! \brief Smaller than comparison operator */
    bool operator<( const VectorGeneric<K, T> &q )const{
        if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]<q[0] ? true : false;
        if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]<q[1] ? true : false;
        return fabs(vals[2]-q[2])>EPSILON && vals[2]<q[2];
    }

    /*! \brief Larger than comparison operator */
    bool operator>( const VectorGeneric<T, K> &q )const{
        if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]>q[0] ? true : false;
        if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]>q[1] ? true : false;
        return fabs(vals[2]-q[2])>EPSILON && vals[2]>q[2];
    }

    /*! \brief Larger than comparison operator */
    bool operator>( const VectorGeneric<K, T> &q )const{
        if( fabs(vals[0]-q[0])>EPSILON ) return vals[0]>q[0] ? true : false;
        if( fabs(vals[1]-q[1])>EPSILON ) return vals[1]>q[1] ? true : false;
        return fabs(vals[2]-q[2])>EPSILON && vals[2]<q[2];
    }

    /*! \brief Equality comparison operator */
    bool operator==( const VectorGeneric<T, K> &q )const{
        return fabs(vals[0]-q[0])<=EPSILON && fabs(vals[1]-q[1])<=EPSILON && fabs(vals[2]-q[2])<=EPSILON;
    }

    /*! \brief Equality comparison operator */
    bool operator==( const VectorGeneric<K, T> &q )const{
        return fabs(vals[0]-q[0])<=EPSILON && fabs(vals[1]-q[1])<=EPSILON && fabs(vals[2]-q[2])<=EPSILON;
    }

    /*! \brief Non-equal comparison operator */
    bool operator!=( const VectorGeneric<T, K> &q )const{
        return fabs(vals[0]-q[0])>EPSILON || fabs(vals[1]-q[1])>EPSILON || fabs(vals[2]-q[2])>EPSILON;
    }

    /*! \brief Non-equal comparison operator */
    bool operator!=( const VectorGeneric<K, T> &q )const{
        return fabs(vals[0]-q[0])>EPSILON || fabs(vals[1]-q[1])>EPSILON || fabs(vals[2]-q[2])>EPSILON;
    }

    /*! \brief Dot product with another vector */
    num dot( const VectorGeneric<T, K> &q )const{
        return vals[0]*q[0]+vals[1]*q[1]+vals[2]*q[2];
    }

    /*! \brief Dot product with another vector */
    num dot( const VectorGeneric<K, T> &q )const{
        return vals[0]*q[0]+vals[1]*q[1]+vals[2]*q[2];
    }

    /*! \brief Cross product with another vector */
    VectorGeneric<T, K> cross( const VectorGeneric<T, K> &q )const{
        return VectorGeneric<T, K>( vals[1]*q[2]-vals[2]*q[1],vals[2]*q[0]-vals[0]*q[2],vals[0]*q[1]-vals[1]*q[0] );
    }

    /*! \brief Cross product with another vector */
    VectorGeneric<num, int> cross( const VectorGeneric<K, T> &q )const{
        return VectorGeneric<num, int>( vals[1]*q[2]-vals[2]*q[1],vals[2]*q[0]-vals[0]*q[2],vals[0]*q[1]-vals[1]*q[0] );
    }

    /*! \brief Length of vector */
    num len()const{
        return sqrt((num) vals[0]*vals[0]+vals[1]*vals[1]+vals[2]*vals[2]);
    }

    /*! \brief Squared length of vector */
    num lenSqr()const{
        return vals[0]*vals[0]+vals[1]*vals[1]+vals[2]*vals[2];
    }

    /*! \brief Distance between two points
     *
     * The points are specified by this and by the q vector.
     */
    num dist( const VectorGeneric<T, K> &q )const{
        num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return sqrt(dx*dx+dy*dy+dz*dz);
    }

    /*! \brief Distance between two points
     *
     * The points are specified by this and by the q vector.
     */
    num dist( const VectorGeneric<K, T> &q )const{
        num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return sqrt(dx*dx+dy*dy+dz*dz);
    }

    /*! \brief Python wrapped dist function */
    num distPython( const VectorGeneric<T, K> &q)const{
        num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return sqrt(dx*dx+dy*dy+dz*dz);
    }

    /*! \brief Squared distance between two points */
    num distSqr( const VectorGeneric<T, K> &q) {
        num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return dx*dx+dy*dy+dz*dz;
    }

    /*! \brief Squared distance between two points */
    num distSqr( const VectorGeneric<K, T> &q) {
        num dx=vals[0]-q[0],dy=vals[1]-q[1],dz=vals[2]-q[2];return dx*dx+dy*dy+dz*dz;
    }

    /*! \brief Return normalized form of this vector */
    VectorGeneric<num, int> normalized()const{
        num l=len();return VectorGeneric<num, int>( vals[0]/l,vals[1]/l,vals[2]/l );
    }

    /*! \brief Normalize this vector */
    void normalize(){
        num l=len();vals[0]/=l;vals[1]/=l;vals[2]/=l; //will not necessarily normalize int vectors
    }

    /*! \brief Mirror vector along y direction */
    VectorGeneric<T, K> perp2d() {
        return VectorGeneric(vals[1], -vals[0], vals[2]);
    }

    /*! \brief Set all elements zero
     *
     * \todo This function is identical to the zero() function
     */
    void clear(){
        vals[0] = vals[1] = vals[2] = 0;
    }

    /*! \brief Convert vector to string for output */
    string asStr() const {
        ostringstream oss;
        oss << "x: " << vals[0] << ", y: " << vals[1] << ", z: " << vals[2];
        return oss.str();
    }

    /*! \brief Assignment operator */
    VectorGeneric<T, K> &operator=(const VectorGeneric<K, T> &other) {
        for (int i=0; i<3; i++) {
            vals[i] = other[i];
        }
        return *this;
    }

    /*! \brief Assignment operator
     *
     * \todo This function is already implicitly defined via the float3
     *       constructor
     */
    VectorGeneric<T, K> &operator=(const float3 &other) {
        vals[0] = other.x;
        vals[1] = other.y;
        vals[2] = other.z;
        return *this;
    }

    /*! \brief Assignment operator
     *
     * \todo This function is already implicitly defined via the float4
     *       constructor
     */
    VectorGeneric<T, K> &operator=(const float4 &other) {
        vals[0] = other.x;
        vals[1] = other.y;
        vals[2] = other.z;
        return *this;
    }

    /*! \brief Copy constructor */
    VectorGeneric<T, K> (const VectorGeneric<K, T> &other) {
        for (int i=0; i<3; i++) {
            vals[i] = other[i];
        }
    }

    /*! \brief Get a specific element */
    T get(int i) {
        return vals[i];
    }

    /*! \brief Set a specific element */
    void set(int i, T val) {
        vals[i] = val;
    }

    /*! \brief I have no idea
     *
     * \todo Someone explain to me what this function does? And write this
     *       documentation please.
     */
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
