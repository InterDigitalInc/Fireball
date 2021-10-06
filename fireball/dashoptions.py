# Copyright (c) 2016-2020 InterDigital AI Research Lab
"""
This file contains the implementation for "DashOption" utility class for handling python command-line arguments.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 11/17/2016    Shahab Hamidi-Rad       First version of the file.
# 12/14/2016    Shahab                  Added support for automatic handling of new
#                                       lines in the description of the options.
# **********************************************************************************************************************
import sys

# ************************************************************************************************
class DashOption:
    """
        __init__
    
    """
    def __init__(self, name, desc, specs=['bool']):
        self.name = name
        self.desc = desc
        if 'string' in specs:       self.optionType = 'string'
        elif 'float' in specs:      self.optionType = 'float'
        elif 'int' in specs:        self.optionType = 'int'
        elif 'bool' in specs:       self.optionType = 'bool'
        else:
            print("DashOption Error(%s): You must specify the type of option. (string, float, int, or bool)"%(name))
            return
        
        self.required = ('required' in specs)
        self.min = None
        self.max = None
        self.validStrs = None
        self.noValDef = None
        self.default = None
        for spec in specs:
            parts = spec.split(':')
            if len(parts)==1:
                if spec not in ['string', 'float', 'int', 'bool', 'required']:
                    print("DashOption Warning(%s): The keyword '%s' not supported! (Ignored)"%(name, spec))
                continue

            if self.optionType in ['float', 'int']:
                specVal = None
                if parts[0] in ['min', 'max', 'default', 'noValDef']:
                    if self.optionType == 'float':
                        try:
                            specVal = float(parts[1])
                        except ValueError:
                            print("DashOption Warning(%s): You need to specify a real number for '%s'! (Ignored)"%(name, parts[0]))
                            continue
                    else:
                        try:
                            specVal = int(parts[1])
                        except ValueError:
                            print("DashOption Warning(%s): You need to specify an integer number for '%s'! (Ignored)"%(name, parts[0]))
                            continue
                    if parts[0] == 'max':           self.max = specVal
                    elif parts[0] == 'min':         self.min = specVal
                    elif parts[0] == 'default':     self.default = specVal
                    elif parts[0] == 'noValDef':    self.noValDef = specVal
                else:
                    print("DashOption Warning(%s): The '%s' keyword is not supported for '%s' values! (Ignored)"%(name, parts[0], self.optionType))
        
            elif self.optionType == 'bool':
                print("DashOption Warning(%s): The '%s' keyword is not supported for '%s' values! (Ignored)"%(name, parts[0], self.optionType))
            elif self.optionType == 'string':
                if parts[0] == 'oneOf':
                    self.validStrs = ':'.join(parts[1:]).split(',')
                    self.validStrs = [ x.strip(' ') for x in self.validStrs]
                elif parts[0] == 'default':     self.default = ':'.join(parts[1:])
                elif parts[0] == 'noValDef':    self.noValDef = ':'.join(parts[1:])
                else:
                    print("DashOption Warning(%s): The '%s' keyword is not supported for '%s' values! (Ignored)"%(name, parts[0], self.optionType))

        if self.optionType == 'bool':
            self.noValDef = True
            self.default = False
        
        self.value = self.default

    # ************************************************************************************************
    def verifyValue(self, optionValue):
        if optionValue is None:
            if self.noValDef is None:
                # This means a value is needed but missing
                return "You need to specify a value for '%s'!"%(self.name)
            self.value = self.noValDef
            return ''

        if self.optionType == 'string':
            if self.validStrs is not None:
                if optionValue not in self.validStrs:
                    return "'%s' must be one of: (%s)!"%(self.name, ', '.join(self.validStrs))
            
            self.value = optionValue
            return ''

        if self.optionType == 'int':
            try:                intValue = int(optionValue)
            except ValueError:  return "You need to specify an integer number for '%s'!"%(self.name)

            if self.min is not None:
                if intValue < self.min:
                    return "'%s' cannot be less than %d!"%(self.name, self.min)
            if self.max is not None:
                if intValue > self.max:
                    return "'%s' cannot be more than %d!"%(self.name, self.max)
            self.value = intValue
            return ''

        if self.optionType == 'float':
            try:                floatValue = float(optionValue)
            except ValueError:  return "You need to specify a real number for '%s'!"%(self.name)
            if self.min is not None:
                if floatValue < self.min:
                    return "'%s' must be more than %f!"%(self.name, self.min)
            if self.max is not None:
                if floatValue > self.max:
                    return "'%s' must be less than %f!"%(self.name, self.max)
            self.value = floatValue
            return ''

        if self.optionType == 'bool':
            self.value = True
            return ''

        return "Internal Error while handling '%s'!!!"%(self.name)
    
# ************************************************************************************************
class DashOptions:
    def __init__(self, options = None):
        self.requireds = {}
        self.optionals = {}
        self.optionNames = []
        if options is not None:
            for optionName, optionDesc, optionSpecs in options:
                self.addOption( optionName, optionDesc, optionSpecs )

    # ************************************************************************************************
    def addOption(self, optionName, optionDesc, optionSpecs):
        if (optionName in self.requireds) or (optionName in self.optionals):
            print("DashOptions Error(%s): This option already exists!"%(optionName))
            return False

        newOption = DashOption(optionName, optionDesc, optionSpecs)
        if newOption.optionType is None:
            return False

        if newOption.required:  self.requireds[ optionName ] = newOption
        else:                   self.optionals[ optionName ] = newOption
        self.optionNames.append( optionName )

    # ************************************************************************************************
    def get(self, optionName):
        if optionName in self.requireds:    return self.requireds[ optionName ].value
        if optionName in self.optionals:    return self.optionals[ optionName ].value
        return None
    
    # ************************************************************************************************
    def run(self, allArgs=None, showUsageOnFail=True):
        if allArgs is None: allArgs = sys.argv
        missingRequiredOptions = set( self.requireds.keys() )
        seenOptions = set()
        for i, arg in enumerate(allArgs):
            if i<1:     continue
            if len(arg)<2:
                print("Invalid Option '%s'!"%(arg))
                if showUsageOnFail: self.showUsage()
                return False

            if arg[0] != '-':
                print("Missing '-' for Option \"%s\"!"%(arg))
                if showUsageOnFail: self.showUsage()
                return False
            
            if arg[1] == '-':   optionParts = arg[2:].split('=')
            else:               optionParts = arg[1:].split('=')
            optionValue = None
            if len(optionParts)>=2:    optionValue = '='.join(optionParts[1:])
            optionName = optionParts[0]
            
            if optionName in seenOptions:
                print("Multiple specifications for option '%s'!"%(optionName))
                if showUsageOnFail: self.showUsage()
                return False
            seenOptions.add( optionName )

            if optionName in ['help', 'h']:
                if showUsageOnFail: self.showUsage()
                return False

            if optionName in self.requireds:
                missingRequiredOptions.remove(optionName)
                errorStr = self.requireds[optionName].verifyValue(optionValue)
                if errorStr != '':
                    print( errorStr )
                    if showUsageOnFail: self.showUsage()
                    return False
                continue

            if optionName in self.optionals:
                errorStr = self.optionals[optionName].verifyValue(optionValue)
                if errorStr != '':
                    print( errorStr )
                    if showUsageOnFail: self.showUsage()
                    return False
                continue
            
            print("Invalid Option '%s'!"%(optionName))
            if showUsageOnFail: self.showUsage()
            return False
        
        if len(missingRequiredOptions)>0:
            print("Missing required parameters:")
            for optionName in missingRequiredOptions:
                print("    %s"%(optionName))
            if showUsageOnFail: self.showUsage()
            return False
        
        return True

    # ************************************************************************************************
    def __repr__(self):
        repStr = '\nCurrent Option Values:\n'
        if len(self.requireds)>0:
            repStr += '    Required:\n'
            for optionName in self.optionNames:
                if optionName not in self.requireds: continue
                if self.requireds[optionName].desc is None: continue
                repStr += '        %s: %s\n'%(optionName, self.requireds[optionName].value)
    
        if len(self.optionals)>0:
            repStr += '    Optional:\n'
            for optionName in self.optionNames:
                if optionName not in self.optionals: continue
                if self.optionals[optionName].desc is None: continue
                repStr += '        %s: %s\n'%(optionName, self.optionals[optionName].value)
        return repStr
                
    # ************************************************************************************************
    def showUsage(self):
        myName = 'python ' + sys.argv[0]
        if 'myPyName' in self.optionNames:
            myName = self.optionals['myPyName'].value
        print('\nUsage: ' + myName + ' [-optionName[=optionValue]]')
        if len(self.requireds)>0:
            print( '    Required:')
            for optionName in self.optionNames:
                if optionName not in self.requireds: continue
                if self.requireds[optionName].desc is None: continue
                desc = self.requireds[optionName].desc.replace('\n', '\n'+13*' ')
                print( '        -%s: %s'%(optionName, desc) )

        if len(self.optionals)>0:
            print( '    Optional:')
            for optionName in self.optionNames:
                if optionName not in self.optionals: continue
                if self.optionals[optionName].desc is None: continue
                desc = self.optionals[optionName].desc.replace('\n', '\n'+13*' ')
                print( '        -%s: %s'%(optionName, desc))
    
        print( '\n')

# ************************************************************************************************
if __name__ == '__main__':
    print('Testing...')
    myOptions = DashOptions(
                             [('intOp1',
                               'Sample Int Option1: min=1, max=10, noValDef=-1, default=3',
                               ['int', 'min:1', 'max:10', 'noValDef:-1', 'default:3']),
                              
                              ('intOp2',
                               'Sample Int Option2: min=1, default=3',
                               ['int', 'min:1', 'default:3']),

                              ('intOp3',
                               'Sample Int Option3: Any int less than 10, A value is needed',
                               ['int', 'max:10']),

                              ('intOp4',
                               'Sample Int Option4: Any int. A value needed. A required option. No Default',
                               ['int', 'required']),
                              
                              ('floatOp5',
                               'Sample float Option5: Any float less than 10, noValDef:-1.2, default:5.5',
                               ['float', 'max:10', 'noValDef:-1.2', 'default:5.5']),

                              ('boolOp6',
                               'Sample bool Option6.',
                               ['bool']),

                              ('stringOp7',
                               'Sample string Option7, oneOf:STR1,STR2,STR3, noValDef:STR4, default:STR5',
                               ['string', 'oneOf:STR1,STR2,STR3', 'noValDef:STR4', 'default:STR5']),

                              ('stringOp8',
                               'Sample Description with new lines\nand another line\nand last!',
                               ['string', 'oneOf:STR1,STR2,STR3', 'noValDef:STR4', 'default:STR5']),
                              
                            ])

    print('Command: python DashOption.py -help')
    if myOptions.run(['DashOption.py','-help'])==False:
        print('Failed!!!!')

    print('Command: python DashOption.py')
    if myOptions.run(['DashOption.py'])==False:
        print('Failed!!!!')

    print('\nCommand: python DashOption.py -intOp1=2')
    if myOptions.run(['DashOption.py', '-intOp1=2'])==False:
        print('Failed!!!!')

    print('\nCommand: python DashOption.py -intOp1=2 -intOp4')
    if myOptions.run(['DashOption.py', '-intOp1=2', '-intOp4'])==False:
        print('Failed!!!!')

    print('\nCommand: python DashOption.py -intOp1=2 -intOp4=1')
    if myOptions.run(['DashOption.py', '-intOp1=2', '-intOp4=1'])==False:
        print('Failed!!!!')
    else:
        print('Values:')
        print('    intOp1: ' + str( myOptions.get('intOp1')))
        print('    intOp2: ' + str( myOptions.get('intOp2')))
        print('    intOp3: ' + str( myOptions.get('intOp3')))
        print('    intOp4: ' + str( myOptions.get('intOp4')))
        print('    floatOp5: ' + str( myOptions.get('floatOp5')))
        print('    boolOp6: ' + str( myOptions.get('boolOp6')))
        print('    stringOp7: ' + str( myOptions.get('stringOp7')))

    print('\nCommand: python DashOption.py -intOp1=12 -intOp4=1')
    if myOptions.run(['DashOption.py', '-intOp1=12', '-intOp4=1'])==False:
        print('Failed!!!!')

    print('\nCommand: python DashOption.py -intOp1=-3 -intOp4=1')
    if myOptions.run(['DashOption.py', '-intOp1=-3', '-intOp4=1'])==False:
        print('Failed!!!!')

    print('\nCommand: python DashOption.py -intOp1=df3 -intOp4=1')
    if myOptions.run(['DashOption.py', '-intOp1=df3', '-intOp4=1'])==False:
        print('Failed!!!!')

    print('\nCommand: python DashOption.py -intOp1=3.5 -intOp4=1')
    if myOptions.run(['DashOption.py', '-intOp1=3.5', '-intOp4=1'])==False:
        print('Failed!!!!')

    print('\nCommand: python DashOption.py -intOp1 -stringOp7=STR1 -intOp4=1')
    if myOptions.run(['DashOption.py', '-intOp1', '-stringOp7=STR1', '-intOp4=1'])==False:
        print('Failed!!!!')
    else:
        print('Values:')
        print('    intOp1: ' + str( myOptions.get('intOp1')))
        print('    intOp2: ' + str( myOptions.get('intOp2')))
        print('    intOp3: ' + str( myOptions.get('intOp3')))
        print('    intOp4: ' + str( myOptions.get('intOp4')))
        print('    floatOp5: ' + str( myOptions.get('floatOp5')))
        print('    boolOp6: ' + str( myOptions.get('boolOp6')))
        print('    stringOp7: ' + str( myOptions.get('stringOp7')))

