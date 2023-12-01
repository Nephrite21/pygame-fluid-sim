import numpy as np
import time
import pygame
import math

#########################################################################
#SETTINGS
#########################################################################
#pygame settings
screenSize = [800,450]

#simulation settings
boundsSize = (600, 350)
gravity = 10
collisionDamping = 1
targetDensity = 1.5
pressureMultiplier = 5
deltaTimeSetting = "fixed" #clock and fixed. use fixed when computing speed is low
deltaTime = 0.16 #seconds per frame

#particle setting
    #particle align setting
particleNumber = 1200
particleSize = 3
particleDistance = 10 # distance between particle's starting point
particleNumberInRow = 30
    #particle physics setting
particleMass = 1
smoothingRadius = 15

#DrawSetting
backgroundColor = (0,0,0)
boundBorderColor = (255,0,0)
particleColor = (255,255,255)

#gradient Setting
stepSize = 0.001
rightVec = np.array([1,0])
downVec = np.array([0,1])


#########################################################################
#CONFIGURATION
#########################################################################
#pygame configuration
screen = pygame.display.set_mode(screenSize)
running = True
clock = pygame.time.Clock()

#simulation configuration
boundsPosition = ((screenSize[0]-boundsSize[0])/2,(screenSize[1]-boundsSize[1])/2)
Center = (screenSize[0]/2,screenSize[1]/2)
halfboundSize = (boundsSize[0]/2 - particleSize, boundsSize[1]/2-particleSize)
if deltaTimeSetting == "clock":
    dt = clock.tick(60)/100
elif deltaTimeSetting == "fixed":
    dt = deltaTime

#particle configuration
    #particle placing
if particleNumber > particleNumberInRow:
    particlePlacementWidth = (particleNumberInRow-1) * particleDistance
else:
    particlePlacementWidth = (particleNumber-1)* particleDistance
particlePlacementHeight = (particleNumber//particleNumberInRow)*particleDistance
particlePlacementStartingPoint = (Center[0] - particlePlacementWidth/2,Center[1] - particlePlacementHeight/2)
gg=[]
hh=[]
for i in range(0,particleNumber):
    gg.append([particlePlacementStartingPoint[0] + (i%particleNumberInRow)*particleDistance,
                     particlePlacementStartingPoint[1] + (i//particleNumberInRow)*particleDistance])
    hh.append([0.0,0.0])
position = np.array(gg)
velocity = np.array(hh)
particleProperties = np.zeros(particleNumber)
particleDensities = np.zeros(particleNumber)
pressureForce = np.zeros((particleNumber,2))
spatialLookup = np.zeros(particleNumber)
startIndices = np.zeros(particleNumber)
smoothingRadiusX2 = smoothingRadius*smoothingRadius
#바보같이 써놓은 코드, 나중에 제대로 바꿀것

#particle smoothing variables
scale = 12/(3.141*pow(smoothingRadius,4))
volume = 3.141 * pow(smoothingRadius,4)/6

#functional functions
def magnitude(a,b=np.zeros(2)):
    squaredValue = 0
    for i in a-b:
        squaredValue += i*i
    return math.sqrt(squaredValue)
def sqrMagnitude(a,b=np.zeros(2)):
    squaredValue = 0
    for i in a-b:
        squaredValue += i*i
    return squaredValue

def normVec2(a):
    if a[0] == 0 and a[1] == 0:
        return np.zeros(2)
    else:
        size = math.sqrt(a[0]*a[0] + a[1]*a[1])
        b= [a[0]/size , a[1]/size]
        return np.array(b)


##################################################################################################
##################################################################################################
##################################################################################################
def SmoothingKernelDerivates(dst):
    if dst >= smoothingRadius:
        return 0.0
    return (dst-smoothingRadius)*scale
def CalculateSharedPressure(densityA, densityB):
    return ((densityA - targetDensity) * pressureMultiplier+(densityB - targetDensity) * pressureMultiplier)/2
def CalculateDensity(samplePoint):
    a= position - samplePoint
    dstX2 = np.power(a[:,0],2) + np.power(a[:,1],2) #distance squared
    dst = np.clip(smoothingRadius-np.sqrt(dstX2), a_min=0.0, a_max=None)
    influence = np.power(dst,2)/volume
    densities = particleMass*influence
    return np.sum(densities)
def CalculatePressureForce(particleIndex):
    pressureForce = np.array([0.0,0.0])
    for i in range(0,particleNumber):
        if i == particleIndex:
            continue
        offset = position[i] - position[particleIndex]
        dst = magnitude(offset)
        if dst == 0:
            dir = np.random.normal(2)
        else:
            dir = offset/dst
        slope = SmoothingKernelDerivates(dst)
        sharedPressure = CalculateSharedPressure(particleDensities[i], particleDensities[particleIndex])
        pressureForce += -sharedPressure * dir * slope * particleMass / particleDensities[i]
    return pressureForce
##################################################################################################
##################################################################################################




def ApplyPressureToParticle_scalar():
    for i in range(particleNumber):
        pressureForce[i] = CalculatePressureForce(i)
    for i in range(particleNumber):
        pressureAcceleration = pressureForce[i] / particleDensities[i]
        velocity[i] += pressureAcceleration * dt

# 벡터화된 함수 정의
def ApplyPressureToParticle_vectorized():
    global pressureForce, velocity
    pressureForce = np.array([CalculatePressureForce(i) for i in range(particleNumber)])
    velocity += (pressureForce.T / particleDensities).T * dt

# 테스트 및 시간 측정

start_time_scalar = time.time()
ApplyPressureToParticle_scalar()
scalar_time = time.time() - start_time_scalar

start_time_vectorized = time.time()
ApplyPressureToParticle_vectorized()
vectorized_time = time.time() - start_time_vectorized

print(f"Scalar time: {scalar_time} seconds")
print(f"Vectorized time: {vectorized_time} seconds")